import pytest

from src.retrieval.rerank import rerank


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, payload):
        self._payload = payload
        self.calls = []

    def post(self, url, json, headers):
        self.calls.append((url, json, headers))
        return _FakeResponse(self._payload)

    def close(self):
        pass


def test_rerank_reorders_by_relevance_score():
    candidates = [(1, "low relevance", 0.9), (2, "high relevance", 0.5)]
    fake_client = _FakeClient({
        "results": [
            {"index": 1, "relevance_score": 0.95},
            {"index": 0, "relevance_score": 0.2},
        ]
    })
    result = rerank("query", candidates, top_n=2, api_key="test-key", client=fake_client)
    assert result == [(2, "high relevance", 0.95), (1, "low relevance", 0.2)]


def test_rerank_empty_candidates_returns_empty_without_calling_api():
    fake_client = _FakeClient({"results": []})
    assert rerank("query", [], api_key="test-key", client=fake_client) == []
    assert fake_client.calls == []


def test_rerank_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    with pytest.raises(ValueError):
        rerank("query", [(1, "a", 1.0)])
