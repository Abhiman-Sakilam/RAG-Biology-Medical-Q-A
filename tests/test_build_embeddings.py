import pytest

from scripts.build_embeddings import build_and_persist


@pytest.fixture(autouse=True)
def _voyage_api_key(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self):
        self.calls = []

    def post(self, url, json, headers):
        self.calls.append((url, json, headers))
        embeddings = {"alpha": [1.0, 0.0], "beta": [0.0, 1.0]}
        data = [
            {"embedding": embeddings[text], "index": i}
            for i, text in enumerate(json["input"])
        ]
        return _FakeResponse({"data": data})

    def close(self):
        pass


def _fake_client():
    return _FakeClient()


def test_build_and_persist_creates_index(tmp_path):
    corpus = [{"id": 1, "passage": "alpha"}, {"id": 2, "passage": "beta"}]
    built = build_and_persist(corpus, tmp_path, "hash-v1", client=_fake_client())
    assert built is True
    assert (tmp_path / "index.faiss").exists()
    assert (tmp_path / "corpus_ids.json").exists()
    assert (tmp_path / "corpus.sha256").read_text().strip() == "hash-v1"


def test_build_and_persist_skips_when_hash_matches(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "corpus.sha256").write_text("hash-v1")
    client = _fake_client()
    built = build_and_persist([{"id": 1, "passage": "alpha"}], tmp_path, "hash-v1", client=client)
    assert built is False
    assert client.calls == []
