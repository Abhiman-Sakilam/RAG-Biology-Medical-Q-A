import numpy as np
import pytest
import faiss

from src.retrieval.dense import (
    embed_texts,
    build_dense_index,
    dense_search,
    save_index,
    load_index,
)


class _FakeEmbeddingItem:
    def __init__(self, embedding):
        self.embedding = embedding


class _FakeEmbeddingResponse:
    def __init__(self, embeddings):
        self.data = [_FakeEmbeddingItem(e) for e in embeddings]


class _FakeEmbeddingClient:
    def __init__(self, embedding_map):
        self.embedding_map = embedding_map
        self.embeddings = self

    def create(self, model, input):
        return _FakeEmbeddingResponse([self.embedding_map[text] for text in input])


def test_embed_texts_normalizes_vectors():
    client = _FakeEmbeddingClient({"a": [3.0, 4.0]})
    vectors = embed_texts(["a"], client=client)
    assert vectors.shape == (1, 2)
    norm = float(np.linalg.norm(vectors[0]))
    assert abs(norm - 1.0) < 1e-6


def test_build_dense_index_and_dense_search_returns_best_match():
    corpus = [{"id": 1, "passage": "alpha"}, {"id": 2, "passage": "beta"}]
    client = _FakeEmbeddingClient({
        "alpha": [1.0, 0.0],
        "beta": [0.0, 1.0],
        "query like alpha": [1.0, 0.0],
    })
    index, built_corpus, id_to_passage = build_dense_index(corpus, client=client)
    results = dense_search("query like alpha", index, built_corpus, id_to_passage, k=1, client=client)
    assert results[0][0] == 1
    assert results[0][1] == "alpha"


def test_save_and_load_index_round_trip(tmp_path):
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    index = faiss.IndexFlatIP(2)
    index.add(vectors)
    corpus = [{"id": 10, "passage": "a"}, {"id": 20, "passage": "b"}]
    save_index(index, corpus, tmp_path)
    loaded = load_index(tmp_path, corpus)
    assert loaded.ntotal == 2


def test_load_index_raises_on_corpus_mismatch(tmp_path):
    vectors = np.array([[1.0, 0.0]], dtype="float32")
    index = faiss.IndexFlatIP(2)
    index.add(vectors)
    save_index(index, [{"id": 10, "passage": "a"}], tmp_path)
    with pytest.raises(ValueError):
        load_index(tmp_path, [{"id": 999, "passage": "different"}])


def test_load_index_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_index(tmp_path, [{"id": 1, "passage": "a"}])
