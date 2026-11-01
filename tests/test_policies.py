"""test_policies.py - Unit tests for EcoBrake intervention policy engine.
"""
import os
from policies import PolicyEngine, POLICY_A_NOTE, POLICY_C_NOTE


def test_policy_a_bounded_retry():
    original_messages = [{"role": "user", "content": "Compute sum"}]
    engine = PolicyEngine("run_test_1", original_messages)
    
    action, retry_messages = engine.evaluate(
        signal="REPETITION",
        checkpoint_text="The answer is ",
        reason="Repeated sentence",
    )
    assert action == "retry"
    assert engine.attempt == 1
    # Checkpoint text retained as assistant message
    assert retry_messages[-2] == {"role": "assistant", "content": "The answer is "}
    # System guidance note appended
    assert retry_messages[-1] == {"role": "system", "content": POLICY_A_NOTE}


def test_policy_b_clean_stop():
    original_messages = [{"role": "user", "content": "Loop again"}]
    engine = PolicyEngine("run_test_2", original_messages)
    
    # First intervention: retry
    act1, _ = engine.evaluate("TOOL_REPEAT")
    assert act1 == "retry"
    
    # Second intervention (retry also flagged): clean stop
    act2, reason = engine.evaluate("TOOL_REPEAT")
    assert act2 == "clean_stop"
    assert reason == "echobrake_stopped"


def test_policy_c_strategy_note(monkeypatch):
    monkeypatch.setenv("ECOBRAKE_ENABLE_POLICY_C", "true")
    original_messages = [{"role": "user", "content": "Stuck in a loop"}]
    engine = PolicyEngine("run_test_3", original_messages)
    
    action, retry_messages = engine.evaluate("STALLED")
    assert action == "retry"
    assert retry_messages[-1] == {"role": "system", "content": POLICY_C_NOTE}


def test_create_stop_chunk():
    chunk = PolicyEngine.create_stop_chunk("chk-123", "gpt-4o-mini", finish_reason="echobrake_stopped")
    assert 'data: {"id": "chk-123"' in chunk
    assert '"finish_reason": "echobrake_stopped"' in chunk
    assert "data: [DONE]\n\n" in chunk
