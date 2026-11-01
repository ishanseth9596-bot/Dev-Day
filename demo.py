"""demo.py - Send demo requests to the EcoBrake proxy and watch interventions live.
Usage:
  python demo.py
"""
import json
import time
import httpx

PROXY_URL = "http://127.0.0.1:8000"


def run_demo():
    print("=" * 60)
    print(" EcoBrake Live Intervention Demonstration")
    print("=" * 60)
    print("Sending a stalled stream (Sentence Repetition Loop)...")

    with httpx.Client(timeout=30.0) as client:
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Explain background worker convergence"}],
            "replay_file": "bench/stalled_runs/stalled_sentence_loop.json",
            "stream": True,
        }
        resp = client.post(f"{PROXY_URL}/v1/chat/completions", json=payload)
        print("\nStreaming response received through EcoBrake:")
        for line in resp.text.splitlines():
            if line.startswith("data:") and not line.endswith("[DONE]"):
                data_str = line[5:].strip()
                try:
                    obj = json.loads(data_str)
                    choice = obj.get("choices", [{}])[0]
                    delta = choice.get("delta", {}).get("content", "")
                    reason = choice.get("finish_reason")
                    if delta:
                        print(delta, end="", flush=True)
                    if reason:
                        print(f"\n[Intervention Finish Reason: {reason}]")
                except Exception:
                    pass

        time.sleep(0.5)

        stats_resp = client.get(f"{PROXY_URL}/api/stats")
        if stats_resp.status_code == 200:
            stats = stats_resp.json()
            print("\n" + "-" * 60)
            print(" Current Live Dashboard Stats:")
            print(f"  * Total Runs:           {stats.get('runs')}")
            print(f"  * Interventions:        {stats.get('interventions')}")
            print(f"  * Est. Tokens Saved:    {stats.get('estimated_tokens_saved')}")
            print(f"  * Mean Added Latency:   {stats.get('mean_added_latency_ms')} ms")
            print("-" * 60)
            print("[SUCCESS] Check your browser on http://localhost:8000/ to see the live feed update!")


if __name__ == "__main__":
    run_demo()
