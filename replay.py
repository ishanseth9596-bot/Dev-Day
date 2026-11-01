"""replay.py - EcoBrake replay engine for deterministic chunk streaming.
Reads recorded SSE chunk files in JSONL or JSON format and replays them asynchronously.
"""
from __future__ import annotations
import asyncio
import json
import os
from typing import AsyncGenerator, Dict, List, Optional


def load_replay_file(file_path: str) -> List[Dict[str, any]]:
    """Load JSONL or JSON benchmark recorded chunks."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Replay file not found: {file_path}")
    if file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("stream_chunks", [])
    chunks = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


async def stream_from_replay(
    file_path: str, fast_mode: bool = False
) -> AsyncGenerator[str, None]:
    """Yield chunks from recorded file with simulated timing."""
    chunks = load_replay_file(file_path)
    last_ts = 0.0

    for item in chunks:
        chunk_str = item.get("chunk", "")
        ts = item.get("timestamp_ms", 0.0)
        delay = (ts - last_ts) / 1000.0
        last_ts = ts

        if not fast_mode and delay > 0:
            await asyncio.sleep(min(delay, 0.05))

        yield chunk_str


def parse_sse_chunk(raw_chunk: str) -> Optional[Dict[str, any]]:
    """Extract JSON object from SSE data payload if present."""
    for line in raw_chunk.splitlines():
        line = line.strip()
        if line.startswith("data:") and not line.endswith("[DONE]"):
            payload_str = line[5:].strip()
            if payload_str:
                try:
                    return json.loads(payload_str)
                except Exception:
                    pass
    return None
