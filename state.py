"""state.py - EcoBrake runtime state, metrics aggregation, and runs.jsonl logger.
"""
from __future__ import annotations
import json
import os
import time
from typing import Any, Dict, List, Optional


class RuntimeMetrics:
    """Thread-safe in-memory store for proxy runs, interventions, and events."""

    def __init__(self, log_path: str = "runs.jsonl") -> None:
        self.log_path = log_path
        self.total_runs: int = 0
        self.total_interventions: int = 0
        self.false_positive_candidates: int = 0
        self.estimated_tokens_saved: int = 0
        self.total_latency_added_ms: float = 0.0
        self.events: List[Dict[str, Any]] = []
        self.strictness: str = "balanced"
        self.sentence_repeat_threshold: int = 3
        self.tool_repeat_threshold: int = 3
        self.stall_steps: int = 3

    def get_config(self) -> Dict[str, Any]:
        return {
            "strictness": self.strictness,
            "sentence_repeat_threshold": self.sentence_repeat_threshold,
            "tool_repeat_threshold": self.tool_repeat_threshold,
            "stall_steps": self.stall_steps,
        }

    def set_config(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        mode = str(cfg.get("strictness", self.strictness)).lower()
        if mode == "strict":
            self.strictness, self.sentence_repeat_threshold, self.tool_repeat_threshold, self.stall_steps = "strict", 2, 2, 2
        elif mode == "lenient":
            self.strictness, self.sentence_repeat_threshold, self.tool_repeat_threshold, self.stall_steps = "lenient", 5, 4, 5
        else:
            self.strictness, self.sentence_repeat_threshold, self.tool_repeat_threshold, self.stall_steps = "balanced", 3, 3, 3
        if "sentence_repeat_threshold" in cfg:
            self.sentence_repeat_threshold = int(cfg["sentence_repeat_threshold"])
        if "tool_repeat_threshold" in cfg:
            self.tool_repeat_threshold = int(cfg["tool_repeat_threshold"])
        if "stall_steps" in cfg:
            self.stall_steps = int(cfg["stall_steps"])
        return self.get_config()

    def record_run(self) -> None:
        self.total_runs += 1

    def record_intervention(
        self,
        run_id: str,
        signal: str,
        policy: str,
        action: str,
        tokens_before: int,
        tokens_after: int,
        latency_added_ms: float,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record intervention to memory and append to runs.jsonl."""
        self.total_interventions += 1
        # Estimated tokens saved is runaway tokens prevented
        saved = max(0, tokens_before - tokens_after) if policy == "clean_stop" else max(0, 150 - tokens_after)
        self.estimated_tokens_saved += saved
        self.total_latency_added_ms += latency_added_ms

        entry = {
            "run_id": run_id,
            "timestamp": time.time(),
            "signal": signal,
            "policy": policy,
            "action": action,
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "latency_added_ms": round(latency_added_ms, 2),
            "tokens_saved": saved,
            "details": details or {},
        }
        self.events.append(entry)

        # Append to runs.jsonl
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                json_line = json.dumps(
                    {
                        "run_id": entry["run_id"],
                        "timestamp": entry["timestamp"],
                        "signal": entry["signal"],
                        "policy": entry["policy"],
                        "action": entry["action"],
                        "tokens_before": entry["tokens_before"],
                        "tokens_after": entry["tokens_after"],
                        "latency_added_ms": entry["latency_added_ms"],
                    }
                )
                f.write(json_line + "\n")
        except Exception:
            pass

        return entry

    def get_stats(self) -> Dict[str, Any]:
        mean_latency = (
            round(self.total_latency_added_ms / self.total_interventions, 2)
            if self.total_interventions > 0
            else 0.0
        )
        cost_saved = round((self.estimated_tokens_saved / 1_000_000) * 0.60, 5)
        health_score = 100 if self.total_runs == 0 else max(82, int(100 - (self.total_interventions / max(1, self.total_runs)) * 10))
        return {
            "runs": self.total_runs,
            "interventions": self.total_interventions,
            "false_positive_candidates": self.false_positive_candidates,
            "estimated_tokens_saved": self.estimated_tokens_saved,
            "estimated_cost_saved_usd": cost_saved,
            "mean_added_latency_ms": mean_latency,
            "agent_health_score": health_score,
            "strictness": self.strictness,
        }

    def get_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        return list(reversed(self.events[-limit:]))


# Global metrics instance
metrics = RuntimeMetrics()
