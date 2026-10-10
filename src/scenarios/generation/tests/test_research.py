import json
from types import SimpleNamespace

from scenarios.generation import research


def setup(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "private-test-key")
    monkeypatch.setenv("SCENARIO_RESEARCH_LOG", str(tmp_path / "outside/research.jsonl"))
    monkeypatch.setattr(research.time, "sleep", lambda _: None)


def test_authenticated_search_saves_evidence_without_credentials(monkeypatch, tmp_path):
    setup(monkeypatch, tmp_path)
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(status_code=200, json=lambda: {"data": [{"paperId": "paper-1"}]})

    monkeypatch.setattr(research.requests, "get", get)
    result = research.search_papers("transformer condition monitoring")
    assert result["status"] == "ok" and result["authenticated"]
    assert calls[0][0] == research.ENDPOINT
    assert calls[0][1]["headers"]["x-api-key"] == "private-test-key"
    assert calls[0][1]["allow_redirects"] is False
    receipt = json.loads((tmp_path / "outside/research.jsonl").read_text())
    assert receipt["paper_count"] == 1
    assert "private-test-key" not in json.dumps(result)
    assert "private-test-key" not in (tmp_path / "outside/research.jsonl").read_text()
    assert json.loads((tmp_path / result["response_file"]).read_text()) == result


def test_rate_limits_are_bounded_and_recorded(monkeypatch, tmp_path):
    setup(monkeypatch, tmp_path)
    monkeypatch.setattr(research.requests, "get", lambda *a, **k: SimpleNamespace(status_code=429))
    result = research.search_papers("transformer failures")
    assert result["attempts"] == [429, 429, 429]
    assert result["status"] == "error"
    assert (tmp_path / result["response_file"]).exists()


def test_redirects_are_rejected_without_following_or_logging_response(monkeypatch, tmp_path):
    setup(monkeypatch, tmp_path)
    monkeypatch.setattr(research.requests, "get", lambda *a, **k: SimpleNamespace(status_code=302))
    result = research.search_papers("transformer failures")
    assert result["status"] == "error" and result["attempts"] == [302]


def test_network_error_is_recorded_without_exception_details(monkeypatch, tmp_path):
    setup(monkeypatch, tmp_path)

    def fail(*args, **kwargs):
        raise research.requests.RequestException("sensitive transport details")

    monkeypatch.setattr(research.requests, "get", fail)
    result = research.search_papers("transformer failures")
    assert result["status"] == "error"
    assert "sensitive" not in json.dumps(result)
