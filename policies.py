"""policies.py - EcoBrake intervention policy engine.
Implements Policy A (bounded retry), Policy B (clean stop), and Policy C (strategy change note).
"""
from __future__ import annotations
import copy
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

POLICY_A_NOTE = (
    "Previous attempt looped. Continue from the last valid point without repeating earlier content."
)
POLICY_C_NOTE = (
    "Previous approach stalled. Please try an alternative strategy or tool."
)


def is_policy_c_enabled() -> bool:
    return os.getenv("ECOBRAKE_ENABLE_POLICY_C", "false").lower() in ("true", "1", "yes")


class PolicyEngine:
    """Evaluates detector flags and produces intervention actions (retry or clean stop)."""

    def __init__(self, run_id: str, original_messages: List[Dict[str, Any]]) -> None:
        self.run_id = run_id
        self.original_messages = copy.deepcopy(original_messages)
        self.attempt = 0
        self.start_time = time.time()

    def evaluate(
        self,
        signal: str,
        checkpoint_text: Optional[str] = None,
        reason: str = "",
    ) -> Tuple[str, Any]:
        """Returns (action_type, payload).
        action_type is either 'retry' (with new messages) or 'clean_stop' (with finish_reason).
        """
        if self.attempt == 0:
            self.attempt = 1
            # Policy A (or Policy C variant on STALLED)
            retry_messages = copy.deepcopy(self.original_messages)

            note = (
                POLICY_C_NOTE
                if (signal == "STALLED" and is_policy_c_enabled())
                else POLICY_A_NOTE
            )

            if signal == "REPETITION" and checkpoint_text:
                # Include valid prefix before repetition as assistant content
                if checkpoint_text.strip():
                    retry_messages.append({"role": "assistant", "content": checkpoint_text})
            
            # Append intervention system guidance
            retry_messages.append({"role": "system", "content": note})
            return "retry", retry_messages
        else:
            # Policy B: Clean stop
            return "clean_stop", "echobrake_stopped"

    @staticmethod
    def create_stop_chunk(
        chunk_id: str,
        model: str,
        finish_reason: str = "echobrake_stopped",
        content: str = "",
    ) -> str:
        """Create an SSE chunk compatible with OpenAI streaming format."""
        payload = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": content} if content else {},
                    "finish_reason": finish_reason,
                }
            ],
        }
        return f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n"
