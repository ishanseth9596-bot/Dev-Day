"""recorder.py - Record live or mock upstream LLM streaming chunks to replay JSONL.
Usage:
  python recorder.py --output bench/stalled_runs/replay_sample.jsonl --prompt "Repeat"
  python recorder.py --mock-stall sentence --output bench/stalled_runs/rep.jsonl
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import sys
import time
import httpx


def generate_mock_stream(mode: str) -> list[dict]:
    """Generate synthetic chunks for testing without live API keys."""
    chunks = []
    base_id = "chatcmpl-mock"
    t = 0.0

    if mode == "sentence_repeat":
        # Stream text that repeats a 30+ char sentence 4 times
        sentence = "The quick brown fox jumps repeatedly over the lazy dog."
        parts = [
            "Here is the detailed analysis you requested. ",
            f"{sentence} ",
            f"{sentence} ",
            f"{sentence} ",
            f"{sentence} ",
            "Final conclusion reached.",
        ]
        for p in parts:
            t += 20.0
            raw = f"data: {json.dumps({'id': base_id, 'choices': [{'delta': {'content': p}, 'finish_reason': None}]})}\n\n"
            chunks.append({"chunk": raw, "timestamp_ms": round(t, 2)})
    elif mode == "tool_repeat":
        # Assistant emitting 3 identical tool calls
        for i in range(3):
            t += 25.0
            raw = f"data: {json.dumps({'id': base_id, 'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': {'name': 'query_db', 'arguments': '{\"table\": \"users\"}'}}]}, 'finish_reason': None}]})}\n\n"
            chunks.append({"chunk": raw, "timestamp_ms": round(t, 2)})
    elif mode == "healthy":
        # Normal, varied streaming text
        parts = [
            "Hello! I am ready to help you analyze your application logs.\n\n",
            "First, let's inspect the error rates observed over the past hour.\n",
            "The latency metrics appear completely stable across all nodes.\n",
            "Everything looks healthy and operational.",
        ]
        for p in parts:
            t += 20.0
            raw = f"data: {json.dumps({'id': base_id, 'choices': [{'delta': {'content': p}, 'finish_reason': None}]})}\n\n"
            chunks.append({"chunk": raw, "timestamp_ms": round(t, 2)})

    t += 10.0
    chunks.append({
        "chunk": f"data: {json.dumps({'id': base_id, 'choices': [{'delta': {}, 'finish_reason': 'stop'}]})}\n\ndata: [DONE]\n\n",
        "timestamp_ms": round(t, 2),
    })
    return chunks


async def record_live_stream(url: str, key: str, model: str, prompt: str, output: str) -> None:
    """Record chunks from a live OpenAI-compatible endpoint."""
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    endpoint = f"{url.rstrip('/')}/chat/completions"
    start_time = time.time()
    chunks = []

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", endpoint, headers=headers, json=body) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_text():
                rel_ms = (time.time() - start_time) * 1000.0
                chunks.append({"chunk": chunk, "timestamp_ms": round(rel_ms, 2)})

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    print(f"Recorded {len(chunks)} chunks to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record SSE streaming chunks")
    parser.add_argument("--output", required=True, help="Destination JSONL path")
    parser.add_argument("--mock-stall", choices=["sentence_repeat", "tool_repeat", "healthy"], help="Generate mock dataset")
    parser.add_argument("--url", default=os.getenv("UPSTREAM_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--key", default=os.getenv("UPSTREAM_API_KEY", ""))
    parser.add_argument("--model", default=os.getenv("UPSTREAM_MODEL", "gpt-4o-mini"))
    parser.add_argument("--prompt", default="Explain quantum computing briefly.")
    args = parser.parse_args()

    if args.mock_stall:
        chunks = generate_mock_stream(args.mock_stall)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c) + "\n")
        print(f"Generated {len(chunks)} mock chunks to {args.output}")
    else:
        asyncio.run(record_live_stream(args.url, args.key, args.model, args.prompt, args.output))


if __name__ == "__main__":
    main()
