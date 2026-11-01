"""test_ledger.py - Unit tests for EcoBrake agent progress ledger.
"""
from ledger import ProgressLedger


def test_progress_with_new_tool():
    ledger = ProgressLedger(stall_threshold=3)
    flagged, stalls, _ = ledger.record_step(
        tool_keys=["search_web::{\"q\":\"ai\"}"],
        tool_results=["Result content 1"],
    )
    assert not flagged
    assert stalls == 0
    assert len(ledger.steps) == 1
    assert ledger.steps[0].made_progress


def test_progress_with_new_result_and_fact():
    ledger = ProgressLedger(stall_threshold=3)
    # Step 1
    ledger.record_step(["search::1"], ["Result A"], ["Fact: Earth is round"])
    # Step 2: Same tool key, but different result hash
    flagged, stalls, _ = ledger.record_step(["search::1"], ["Result B"], [])
    assert not flagged
    assert stalls == 0

    # Step 3: Same tool key, same result, but new fact
    flagged, stalls, _ = ledger.record_step(["search::1"], ["Result B"], ["Fact: Water is wet"])
    assert not flagged
    assert stalls == 0


def test_stalled_consecutive_steps():
    ledger = ProgressLedger(stall_threshold=3)
    # Initial progressive step
    ledger.record_step(["tool::a"], ["Result 100"], ["Fact A"])
    
    # 3 consecutive steps with identical tool key, identical result, and no new facts
    f1, s1, _ = ledger.record_step(["tool::a"], ["Result 100"], [])
    assert not f1 and s1 == 1

    f2, s2, _ = ledger.record_step(["tool::a"], ["Result 100"], [])
    assert not f2 and s2 == 2

    f3, s3, reason = ledger.record_step(["tool::a"], ["Result 100"], [])
    assert f3 and s3 == 3
    assert "consecutive steps without new tool keys" in reason


def test_check_message_history():
    ledger = ProgressLedger(stall_threshold=3)
    # 1 initial step + 3 consecutive stalled steps
    messages = [
        {"role": "user", "content": "Help me search"},
        # Step 0 (initial progress)
        {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "lookup", "arguments": "{\"id\":1}"}}],
        },
        {"role": "tool", "content": "record not found"},
        # Stalled step 1
        {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "lookup", "arguments": "{\"id\":1}"}}],
        },
        {"role": "tool", "content": "record not found"},
        # Stalled step 2
        {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "lookup", "arguments": "{\"id\":1}"}}],
        },
        {"role": "tool", "content": "record not found"},
        # Stalled step 3 -> reaches threshold of 3 consecutive stalls!
        {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "lookup", "arguments": "{\"id\":1}"}}],
        },
        {"role": "tool", "content": "record not found"},
    ]
    stalled, reason = ledger.check_message_history(messages)
    assert stalled
    assert "consecutive steps" in reason
