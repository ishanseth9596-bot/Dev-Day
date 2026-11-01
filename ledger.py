"""ledger.py - EcoBrake agent progress ledger.
Tracks tool calls, tool results, and structured facts across agent steps to detect stalls.
"""
from __future__ import annotations
import hashlib
import json
from typing import Any, Dict, List, Optional, Set, Tuple


class StepEntry:
    def __init__(
        self,
        step_index: int,
        tool_keys: List[str],
        result_hashes: List[str],
        facts: List[str],
        made_progress: bool,
    ) -> None:
        self.step_index = step_index
        self.tool_keys = tool_keys
        self.result_hashes = result_hashes
        self.facts = facts
        self.made_progress = made_progress


class ProgressLedger:
    """Maintains a run-scoped progress ledger across assistant-tool turns."""

    def __init__(self, stall_threshold: int = 3) -> None:
        self.stall_threshold = stall_threshold
        self.seen_tool_keys: Set[str] = set()
        self.seen_result_hashes: Set[str] = set()
        self.seen_facts: Set[str] = set()
        self.consecutive_stalls: int = 0
        self.steps: List[StepEntry] = []

    @staticmethod
    def hash_result(result_content: Any) -> str:
        """Compute sha256 hash of tool output."""
        if isinstance(result_content, (dict, list)):
            serialized = json.dumps(result_content, sort_keys=True)
        else:
            serialized = str(result_content)
        return hashlib.sha256(serialized.strip().encode("utf-8")).hexdigest()

    @staticmethod
    def normalize_fact(fact: str) -> str:
        return " ".join(fact.strip().lower().split())

    def record_step(
        self,
        tool_keys: List[str],
        tool_results: List[Any],
        facts: Optional[List[str]] = None,
    ) -> Tuple[bool, int, str]:
        """Record an agent step. Returns (flagged, consecutive_stalls, reason)."""
        made_progress = False

        for key in tool_keys:
            if key not in self.seen_tool_keys:
                made_progress = True
                self.seen_tool_keys.add(key)

        res_hashes: List[str] = []
        for res in tool_results:
            h = self.hash_result(res)
            res_hashes.append(h)
            if h not in self.seen_result_hashes:
                made_progress = True
                self.seen_result_hashes.add(h)

        norm_facts: List[str] = []
        if facts:
            for fact in facts:
                nf = self.normalize_fact(fact)
                if nf:
                    norm_facts.append(nf)
                    if nf not in self.seen_facts:
                        made_progress = True
                        self.seen_facts.add(nf)

        step_idx = len(self.steps)
        if made_progress:
            self.consecutive_stalls = 0
        else:
            self.consecutive_stalls += 1

        self.steps.append(
            StepEntry(step_idx, tool_keys, res_hashes, norm_facts, made_progress)
        )

        flagged = self.consecutive_stalls >= self.stall_threshold
        reason = (
            f"{self.consecutive_stalls} consecutive steps without new tool keys, "
            f"new results, or new facts"
            if flagged
            else ""
        )
        return flagged, self.consecutive_stalls, reason

    def check_message_history(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[bool, str]:
        """Scan OpenAI message history for stalled agent turns."""
        self.reset()
        i = 0
        n = len(messages)
        flagged = False
        reason = ""

        while i < n:
            msg = messages[i]
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                keys = []
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    raw_args = fn.get("arguments", "")
                    keys.append(f"{name}::{raw_args}")
                
                # Gather succeeding tool results
                results = []
                j = i + 1
                while j < n and messages[j].get("role") == "tool":
                    results.append(messages[j].get("content", ""))
                    j += 1
                
                facts = msg.get("facts", [])
                flagged, _, reason = self.record_step(keys, results, facts)
                if flagged:
                    return True, reason
                i = j
            else:
                i += 1
        return flagged, reason

    def reset(self) -> None:
        self.seen_tool_keys.clear()
        self.seen_result_hashes.clear()
        self.seen_facts.clear()
        self.consecutive_stalls = 0
        self.steps.clear()
