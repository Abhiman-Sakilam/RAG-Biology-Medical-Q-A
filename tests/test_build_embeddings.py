from unittest import mock

from scripts.build_embeddings import build_and_persist


def _fake_client():
    client = mock.Mock()
    client.embeddings.create.return_value = mock.Mock(
        data=[mock.Mock(embedding=[1.0, 0.0]), mock.Mock(embedding=[0.0, 1.0])]
    )
    return client


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
    client.embeddings.create.assert_not_called()
