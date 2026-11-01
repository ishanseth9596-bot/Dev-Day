"""test_detectors.py - Unit tests for EcoBrake signal detectors.
"""
import pytest
from detectors import SentenceRepetitionDetector, ToolCallTracker, normalize_sentence


def test_sentence_normalization():
    s1 = "  The Quick Brown Fox Jumps Over!  "
    s2 = "the quick   brown fox jumps over."
    assert normalize_sentence(s1) == normalize_sentence(s2)


def test_sentence_repetition_looping():
    detector = SentenceRepetitionDetector(window_size=300, repeat_threshold=3, min_sentence_len=25)
    loop_sentence = "This is a repeating sentence that exceeds twenty five characters."
    
    # Process chunks progressively
    flagged, _, cp = detector.process_chunk("Starting normal generation. ")
    assert not flagged
    
    flagged, _, cp = detector.process_chunk(f"{loop_sentence} ")
    assert not flagged
    
    flagged, _, cp = detector.process_chunk(f"{loop_sentence} ")
    assert not flagged
    
    # Third repetition within 300-char window triggers flag
    flagged, rep_sent, cp = detector.process_chunk(f"{loop_sentence} ")
    assert flagged
    assert "repeating sentence" in rep_sent
    assert cp > 0


def test_legitimate_repeated_structure():
    detector = SentenceRepetitionDetector(window_size=300, repeat_threshold=3, min_sentence_len=25)
    # Short repeated tokens or short phrases (<25 chars) should never flag
    chunks = [
        "Yes. ", "Yes. ", "Yes. ", "Yes. ",
        "item = 1\n", "item = 2\n", "item = 3\n",
        "for x in list:\n", "for y in list:\n"
    ]
    for c in chunks:
        flagged, _, _ = detector.process_chunk(c)
        assert not flagged


def test_tool_call_repetition():
    tracker = ToolCallTracker(repeat_threshold=3)
    
    # Same tool and args (even with different JSON formatting/spacing)
    flagged1, k1, c1 = tracker.record_call("search_web", '{"query": "python", "limit": 10}')
    assert not flagged1 and c1 == 1
    
    flagged2, k2, c2 = tracker.record_call("search_web", '{"limit": 10, "query": "python"}')
    assert not flagged2 and c2 == 2
    assert k1 == k2  # Canonicalized match
    
    flagged3, k3, c3 = tracker.record_call("search_web", {"query": "python", "limit": 10})
    assert flagged3 and c3 == 3


def test_different_tool_calls_healthy():
    tracker = ToolCallTracker(repeat_threshold=3)
    tracker.record_call("search_web", '{"query": "step 1"}')
    tracker.record_call("search_web", '{"query": "step 2"}')
    flagged, _, _ = tracker.record_call("fetch_page", '{"url": "https://example.com"}')
    assert not flagged
