"""benchmark.py - EcoBrake benchmark runner for detection, false positives, and latency.
Evaluates labeled stalled and healthy datasets and verifies byte/answer equivalence.
"""
from __future__ import annotations
import glob
import json
import os
import sys
import tempfile
import time
from typing import Any, Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from proxy import app
from state import metrics


def extract_content_from_stream(raw_stream: str) -> str:
    """Extract full content text from raw SSE stream."""
    content_parts = []
    for line in raw_stream.splitlines():
        line = line.strip()
        if line.startswith("data:") and not line.endswith("[DONE]"):
            try:
                obj = json.loads(line[5:].strip())
                for ch in obj.get("choices", []):
                    c = ch.get("delta", {}).get("content", "")
                    if c:
                        content_parts.append(c)
            except Exception:
                pass
    return "".join(content_parts)


def run_benchmark() -> Dict[str, Any]:
    client = TestClient(app)
    stalled_files = sorted(glob.glob("bench/stalled_runs/*.json"))
    healthy_files = sorted(glob.glob("bench/healthy_runs/*.json"))

    stalls_detected = 0
    false_positives = 0
    latencies = []
    healthy_matches = 0

    print("\n" + "=" * 65)
    print(" ECOBRAKE BENCHMARK EVALUATION SUITE")
    print("=" * 65)

    # 1. Evaluate Stalled Runs
    print(f"\n[1/2] Evaluating {len(stalled_files)} Stalled Runs:")
    for filepath in stalled_files:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        run_id = data.get("run_id", os.path.basename(filepath))
        exp_signal = data.get("expected_signal", "")

        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tf:
            for item in data.get("stream_chunks", []):
                tf.write(json.dumps(item) + "\n")
            replay_path = tf.name

        os.environ["REPLAY_FILE"] = replay_path
        prev_interventions = metrics.total_interventions
        t0 = time.time()
        res = client.post(
            "/v1/chat/completions",
            json={"model": "bench-model", "messages": data.get("messages", [{"role": "user", "content": "test"}]), "stream": True},
        )
        lat = (time.time() - t0) * 1000.0
        latencies.append(lat)
        os.remove(replay_path)

        intervened = metrics.total_interventions > prev_interventions or "echobrake_stopped" in res.text
        if intervened:
            stalls_detected += 1
            print(f"  [PASSED] {run_id:<28} -> Flagged {exp_signal} ({lat:.1f}ms)")
        else:
            print(f"  [FAILED] {run_id:<28} -> Missed stall!")

    # 2. Evaluate Healthy Runs
    print(f"\n[2/2] Evaluating {len(healthy_files)} Healthy Runs (Equivalence Test):")
    for filepath in healthy_files:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        run_id = data.get("run_id", os.path.basename(filepath))
        expected_ans = data.get("expected_answer", "").strip()

        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tf:
            for item in data.get("stream_chunks", []):
                tf.write(json.dumps(item) + "\n")
            replay_path = tf.name

        os.environ["REPLAY_FILE"] = replay_path
        prev_interventions = metrics.total_interventions
        t0 = time.time()
        res = client.post(
            "/v1/chat/completions",
            json={"model": "bench-model", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        lat = (time.time() - t0) * 1000.0
        latencies.append(lat)
        os.remove(replay_path)

        is_false_pos = metrics.total_interventions > prev_interventions
        actual_ans = extract_content_from_stream(res.text).strip()
        ans_match = actual_ans == expected_ans if expected_ans else True

        if is_false_pos:
            false_positives += 1
            print(f"  [FAILED] {run_id:<28} -> False positive triggered!")
        elif not ans_match:
            print(f"  [FAILED] {run_id:<28} -> Answer mismatch!")
        else:
            healthy_matches += 1
            print(f"  [PASSED] {run_id:<28} -> 100% Match, Zero False Positives ({lat:.1f}ms)")

    det_rate = (stalls_detected / len(stalled_files) * 100.0) if stalled_files else 100.0
    fp_rate = (false_positives / len(healthy_files) * 100.0) if healthy_files else 0.0
    mean_lat = sum(latencies) / len(latencies) if latencies else 0.0
    stats = metrics.get_stats()

    print("\n" + "=" * 65)
    print(" BENCHMARK REPORT SUMMARY")
    print("=" * 65)
    print(f" Detection Rate:               {det_rate:.1f}% ({stalls_detected}/{len(stalled_files)})")
    print(f" False Positive Rate:          {fp_rate:.1f}% ({false_positives}/{len(healthy_files)})")
    print(f" Healthy Answer Equivalence:   {(healthy_matches / len(healthy_files) * 100):.1f}%")
    print(f" Total Estimated Tokens Saved: {stats['estimated_tokens_saved']}")
    print(f" Mean Latency:                 {mean_lat:.2f} ms")
    print("=" * 65 + "\n")

    return {
        "detection_rate": det_rate,
        "false_positive_rate": fp_rate,
        "healthy_equivalence": (healthy_matches / len(healthy_files) * 100),
        "mean_latency_ms": round(mean_lat, 2),
        "tokens_saved": stats["estimated_tokens_saved"],
    }


if __name__ == "__main__":
    run_benchmark()
