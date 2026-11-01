"""chat_live.py - Chat live with a real AI model through the EcoBrake proxy.
Usage:
  python chat_live.py --prompt "Hello! Tell me a fun fact."
"""
import argparse
import json
import os
import sys
import httpx

# Load .env file automatically if present
if os.path.exists(".env"):
    with open(".env", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

PROXY_URL = "http://127.0.0.1:8000"


def chat(key: str, model: str, prompt: str):
    print("=" * 60)
    print(" EcoBrake Live AI Model Streaming")
    print(f" Model: {model}")
    print(f" Prompt: {prompt}")
    print("=" * 60)
    print("Response: ", end="", flush=True)

    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }

    try:
        with httpx.Client(timeout=60.0) as client:
            with client.stream("POST", f"{PROXY_URL}/v1/chat/completions", headers=headers, json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
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
        print("\n\n[SUCCESS] Check your live dashboard at http://localhost:8000/ to see the run recorded!")
    except Exception as e:
        print(f"\n[ERROR] Request failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Chat through EcoBrake with a live AI model")
    parser.add_argument("--key", default=os.getenv("UPSTREAM_API_KEY", ""), help="Upstream API key")
    parser.add_argument("--model", default=os.getenv("UPSTREAM_MODEL", "gpt-4o-mini"), help="Model name")
    parser.add_argument("--prompt", default="Explain why the sky is blue in two sentences.", help="User prompt")
    args = parser.parse_args()

    chat(args.key, args.model, args.prompt)


if __name__ == "__main__":
    main()
