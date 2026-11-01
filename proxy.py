"""proxy.py - EcoBrake lightweight streaming proxy.
Monitors LLM streaming responses for loops and stalls, intervening with bounded recovery.
"""
from __future__ import annotations
import os
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from detectors import SentenceRepetitionDetector, ToolCallTracker
from ledger import ProgressLedger
from policies import PolicyEngine
from replay import parse_sse_chunk, stream_from_replay
from state import metrics

app = FastAPI(title="EcoBrake Runtime Proxy")

if os.path.exists(".env"):
    with open(".env", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

UPSTREAM_BASE_URL = os.getenv("UPSTREAM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
UPSTREAM_API_KEY = os.getenv("UPSTREAM_API_KEY", "")
UPSTREAM_MODEL = os.getenv("UPSTREAM_MODEL", "gpt-4o-mini")


async def fetch_upstream_stream(body: Dict[str, Any], auth_header: str = "") -> AsyncGenerator[str, None]:
    api_key = os.getenv("UPSTREAM_API_KEY", "") or auth_header.replace("Bearer ", "").strip()
    replay_path = body.get("replay_file") or os.getenv("REPLAY_FILE")
    if not replay_path and not api_key:
        replay_path = "bench/stalled_runs/stalled_sentence_loop.json"

    if replay_path and os.path.exists(replay_path):
        async for chunk in stream_from_replay(replay_path):
            yield chunk
        return

    base_url = os.getenv("UPSTREAM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    endpoint = f"{base_url}/chat/completions"
    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", endpoint, headers=headers, json=body) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_text():
                yield chunk


async def execute_monitored_run(
    run_id: str,
    body: Dict[str, Any],
    policy_engine: PolicyEngine,
    auth_header: str = "",
) -> AsyncGenerator[str, None]:
    rep_det = SentenceRepetitionDetector(repeat_threshold=metrics.sentence_repeat_threshold)
    tool_det = ToolCallTracker(repeat_threshold=metrics.tool_repeat_threshold)
    full_text = ""
    t_start = time.time()
    flagged_signal = None
    stop_reason = ""
    checkpoint_idx = 0
    model = body.get("model", UPSTREAM_MODEL)

    # Check incoming messages for multi-step progress stall
    ledger = ProgressLedger(stall_threshold=metrics.stall_steps)
    is_stalled, stall_reason = ledger.check_message_history(body.get("messages", []))
    if is_stalled:
        flagged_signal = "STALLED"
        stop_reason = stall_reason
    else:
        try:
            async for raw_chunk in fetch_upstream_stream(body, auth_header=auth_header):
                sse_data = parse_sse_chunk(raw_chunk)
                if sse_data and "choices" in sse_data and sse_data["choices"]:
                    delta = sse_data["choices"][0].get("delta", {})

                    content = delta.get("content")
                    if content:
                        full_text += content
                        is_rep, rep_sent, cp_idx = rep_det.process_chunk(content)
                        if is_rep:
                            flagged_signal = "REPETITION"
                            stop_reason = f"Sentence repetition: '{rep_sent[:30]}...'"
                            checkpoint_idx = cp_idx
                            break

                    for tc in delta.get("tool_calls", []):
                        fn = tc.get("function", {})
                        name, args = fn.get("name"), fn.get("arguments", "")
                        if name and args:
                            is_tr, _, cnt = tool_det.record_call(name, args)
                            if is_tr:
                                flagged_signal = "TOOL_REPEAT"
                                stop_reason = f"Identical tool call ({cnt}x): {name}"
                                break
                    if flagged_signal:
                        break

                yield raw_chunk
        except Exception as exc:
            stop_reason = f"Stream error: {exc}"

    if flagged_signal:
        lat_added = (time.time() - t_start) * 1000.0
        tokens_before = max(1, len(full_text) // 4)
        clean_prefix = full_text[:checkpoint_idx]
        action, action_payload = policy_engine.evaluate(
            flagged_signal, checkpoint_text=clean_prefix, reason=stop_reason
        )

        policy_name = "bounded_retry" if action == "retry" else "clean_stop"
        action_name = "cut_and_retry" if action == "retry" else "cut_stream"

        metrics.record_intervention(
            run_id=run_id,
            signal=flagged_signal,
            policy=policy_name,
            action=action_name,
            tokens_before=tokens_before,
            tokens_after=max(1, len(clean_prefix) // 4),
            latency_added_ms=lat_added,
            details={"reason": stop_reason},
        )

        if action == "retry":
            retry_body = {**body, "messages": action_payload}
            async for retry_chunk in execute_monitored_run(run_id, retry_body, policy_engine, auth_header=auth_header):
                yield retry_chunk
        else:
            yield policy_engine.create_stop_chunk(
                chunk_id=f"stop-{run_id}", model=model, finish_reason="echobrake_stopped"
            )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> StreamingResponse:
    metrics.record_run()
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    body = await request.json()
    auth_header = request.headers.get("Authorization", "")
    policy_engine = PolicyEngine(run_id, body.get("messages", []))
    return StreamingResponse(
        execute_monitored_run(run_id, body, policy_engine, auth_header=auth_header),
        media_type="text/event-stream",
    )


@app.get("/api/stats")
async def get_stats() -> JSONResponse:
    return JSONResponse(metrics.get_stats())


@app.get("/api/events")
async def get_events(limit: int = 50) -> JSONResponse:
    return JSONResponse(metrics.get_events(limit=limit))


@app.get("/api/parameter-suite")
async def parameter_suite() -> JSONResponse:
    from test_parameters import get_parameter_verification
    return JSONResponse(get_parameter_verification())


@app.get("/api/config")
async def get_config() -> JSONResponse:
    return JSONResponse(metrics.get_config())


@app.post("/api/config")
async def update_config(request: Request) -> JSONResponse:
    body = await request.json()
    return JSONResponse(metrics.set_config(body))


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard() -> HTMLResponse:
    for p in [
        os.path.join(os.path.dirname(__file__), "dashboard", "index.html"),
        os.path.join(os.getcwd(), "dashboard", "index.html"),
    ]:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>EcoBrake Dashboard</h1>")
