"""test_proxy.py - Integration and replay tests for EcoBrake proxy.
"""
import json
import pytest
from fastapi.testclient import TestClient
from proxy import app
from state import metrics


@pytest.fixture
def client():
    return TestClient(app)


def test_api_stats_and_events(client):
    res_stats = client.get("/api/stats")
    assert res_stats.status_code == 200
    stats = res_stats.json()
    assert "runs" in stats
    assert "interventions" in stats
    assert "estimated_tokens_saved" in stats

    res_events = client.get("/api/events")
    assert res_events.status_code == 200
    assert isinstance(res_events.json(), list)


def test_dashboard_route(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "EcoBrake Live Dashboard" in res.text


def test_healthy_passthrough_byte_faithful(client, tmp_path, monkeypatch):
    healthy_file = tmp_path / "healthy.jsonl"
    payload1 = {"id": "1", "choices": [{"delta": {"content": "Hello world!"}, "finish_reason": None}]}
    payload2 = {"id": "1", "choices": [{"delta": {}, "finish_reason": "stop"}]}
    chunk_1 = f"data: {json.dumps(payload1)}\n\n"
    chunk_2 = f"data: {json.dumps(payload2)}\n\ndata: [DONE]\n\n"

    with open(healthy_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"chunk": chunk_1, "timestamp_ms": 10.0}) + "\n")
        f.write(json.dumps({"chunk": chunk_2, "timestamp_ms": 20.0}) + "\n")

    monkeypatch.setenv("REPLAY_FILE", str(healthy_file))
    initial_interventions = metrics.total_interventions

    response = client.post(
        "/v1/chat/completions",
        json={"model": "mock-model", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert response.status_code == 200
    stream_content = response.text
    # Byte-faithful passthrough check:
    assert stream_content == f"{chunk_1}{chunk_2}"
    assert metrics.total_interventions == initial_interventions


def test_stalled_replay_intervention(client, tmp_path, monkeypatch):
    stalled_file = tmp_path / "stalled.jsonl"
    rep_sentence = "This sentence will repeat over and over again."
    chunks = []
    for _ in range(4):
        p = {"id": "st-1", "choices": [{"delta": {"content": f"{rep_sentence} "}, "finish_reason": None}]}
        chunks.append(f"data: {json.dumps(p)}\n\n")

    with open(stalled_file, "w", encoding="utf-8") as f:
        for i, c in enumerate(chunks):
            f.write(json.dumps({"chunk": c, "timestamp_ms": i * 15.0}) + "\n")

    monkeypatch.setenv("REPLAY_FILE", str(stalled_file))
    initial_interventions = metrics.total_interventions

    response = client.post(
        "/v1/chat/completions",
        json={"model": "mock-model", "messages": [{"role": "user", "content": "loop"}], "stream": True},
    )
    assert response.status_code == 200
    assert "echobrake_stopped" in response.text
    assert metrics.total_interventions > initial_interventions
