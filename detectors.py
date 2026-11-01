"""detectors.py - EcoBrake streaming signal detectors.
Implements sentence repetition and tool call repetition detection.
"""
from __future__ import annotations
import json
import re
from typing import Dict, List, Optional, Tuple


def normalize_sentence(sentence: str) -> str:
    """Normalize sentence: lowercase, collapse whitespace, strip punctuation edges."""
    collapsed = " ".join(sentence.strip().lower().split())
    # Strip leading/trailing common punctuation for robust comparison
    return collapsed.strip(".!? \t\r\n")


def split_sentences(text: str) -> List[Tuple[str, int]]:
    """Split text into sentences along with their start index in text."""
    # Split on sentence enders (. ! ? newline) while retaining spans
    pattern = re.compile(r'[^.!?\n]+[.!?\n]*')
    results = []
    for match in pattern.finditer(text):
        raw = match.group(0).strip()
        if raw:
            results.append((raw, match.start()))
    return results


class SentenceRepetitionDetector:
    """Monitors streaming text for repeated sentences within a rolling character window."""

    def __init__(
        self,
        window_size: int = 300,
        repeat_threshold: int = 3,
        min_sentence_len: int = 25,
    ) -> None:
        self.window_size = window_size
        self.repeat_threshold = repeat_threshold
        self.min_sentence_len = min_sentence_len
        self.buffer = ""

    def process_chunk(self, chunk: str) -> Tuple[bool, Optional[str], int]:
        """Append chunk and check for repeated sentence.
        Returns (flagged, repeated_sentence, checkpoint_char_idx).
        """
        self.buffer += chunk
        window_start = max(0, len(self.buffer) - self.window_size)
        window = self.buffer[window_start:]

        sentences_with_pos = split_sentences(window)
        counts: Dict[str, List[int]] = {}

        for raw_s, rel_pos in sentences_with_pos:
            norm = normalize_sentence(raw_s)
            if len(norm) >= self.min_sentence_len:
                abs_pos = window_start + rel_pos
                counts.setdefault(norm, []).append(abs_pos)
                if len(counts[norm]) >= self.repeat_threshold:
                    first_occurrence_pos = counts[norm][0]
                    return True, norm, first_occurrence_pos

        return False, None, len(self.buffer)

    def reset(self) -> None:
        self.buffer = ""


class ToolCallTracker:
    """Tracks identical tool calls within a run by (name, canonicalized JSON arguments)."""

    def __init__(self, repeat_threshold: int = 3) -> None:
        self.repeat_threshold = repeat_threshold
        self.call_counts: Dict[str, int] = {}
        self.call_history: List[str] = []

    @staticmethod
    def canonicalize_key(name: str, args_raw: str | dict) -> str:
        """Create deterministic canonical key: 'name::{"arg":1}'."""
        if isinstance(args_raw, dict):
            canon_args = json.dumps(args_raw, sort_keys=True, separators=(",", ":"))
        else:
            try:
                parsed = json.loads(args_raw)
                canon_args = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
            except Exception:
                canon_args = args_raw.strip()
        return f"{name.strip()}::{canon_args}"

    def record_call(self, name: str, args_raw: str | dict) -> Tuple[bool, str, int]:
        """Record tool call and check if threshold reached.
        Returns (flagged, canonical_key, count).
        """
        key = self.canonicalize_key(name, args_raw)
        count = self.call_counts.get(key, 0) + 1
        self.call_counts[key] = count
        self.call_history.append(key)
        flagged = count >= self.repeat_threshold
        return flagged, key, count

    def reset(self) -> None:
        self.call_counts.clear()
        self.call_history.clear()
