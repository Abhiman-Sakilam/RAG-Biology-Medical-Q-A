import httpx
import pytest

from src.retrieval.rerank import rerank


def _chat_payload(content):
    return {"choices": [{"message": {"content": content}}]}


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://openrouter.ai/api/v1/completions")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError(
                f"{self.status_code} error", request=request, response=response
            )

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self._status_code = status_code
        self.calls = []

    def post(self, url, json, headers):
        self.calls.append((url, json, headers))
        return _FakeResponse(self._payload, self._status_code)

    def close(self):
        pass


def test_rerank_reorders_by_relevance_score():
    candidates = [(1, "low relevance", 0.9), (2, "high relevance", 0.5)]
    fake_client = _FakeClient(_chat_payload("1\n0"))
    result = rerank("query", candidates, top_n=2, api_key="test-key", client=fake_client)
    assert result == [(2, "high relevance", 1.0), (1, "low relevance", 0.5)]


def test_rerank_sends_expected_request_format():
    candidates = [(1, "doc one", 0.9), (2, "doc two", 0.5)]
    fake_client = _FakeClient(_chat_payload("0\n1"))
    rerank("my query", candidates, top_n=2, api_key="test-key", client=fake_client)
    url, payload, headers = fake_client.calls[0]
    assert url == "https://openrouter.ai/api/v1/chat/completions"
    assert payload["model"] == "nvidia/llama-nemotron-rerank-vl-1b-v2:free"
    assert payload["temperature"] == 0
    assert "my query" in payload["messages"][0]["content"]
    assert "doc one" in payload["messages"][0]["content"]
    assert headers == {"Authorization": "Bearer test-key"}


def test_rerank_respects_top_n():
    candidates = [
        (1, "a", 0.1),
        (2, "b", 0.2),
        (3, "c", 0.3),
    ]
    fake_client = _FakeClient(_chat_payload("2\n1\n0"))
    result = rerank("query", candidates, top_n=2, api_key="test-key", client=fake_client)
    assert [pid for pid, _, _ in result] == [3, 2]


def test_rerank_parses_indices_with_surrounding_text():
    candidates = [(1, "a", 0.1), (2, "b", 0.2)]
    fake_client = _FakeClient(_chat_payload("Index: 1 - most relevant\n0. less relevant"))
    result = rerank("query", candidates, top_n=2, api_key="test-key", client=fake_client)
    assert [pid for pid, _, _ in result] == [2, 1]


def test_rerank_falls_back_to_original_order_when_unparseable():
    candidates = [(1, "a", 0.1), (2, "b", 0.2)]
    fake_client = _FakeClient(_chat_payload("I cannot rank these documents."))
    result = rerank("query", candidates, top_n=2, api_key="test-key", client=fake_client)
    assert result == candidates[:2]


def test_rerank_empty_candidates_returns_empty_without_calling_api():
    fake_client = _FakeClient(_chat_payload(""))
    assert rerank("query", [], api_key="test-key", client=fake_client) == []
    assert fake_client.calls == []


def test_rerank_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError):
        rerank("query", [(1, "a", 1.0)])


def test_rerank_propagates_429_error():
    candidates = [(1, "a", 0.1)]
    fake_client = _FakeClient(_chat_payload(""), status_code=429)
    with pytest.raises(httpx.HTTPStatusError):
        rerank("query", candidates, api_key="test-key", client=fake_client)
