import json

from src.data.loaders import load_corpus
from src.data.chunking import parent_id_of


def _write_corpus(tmp_path, passages):
    data_dir = tmp_path / "text-corpus"
    data_dir.mkdir()
    with open(data_dir / "train-00000-of-00001.json", "w") as f:
        json.dump(passages, f)
    return data_dir


def test_load_corpus_returns_short_passages_unchanged(tmp_path):
    data_dir = _write_corpus(
        tmp_path,
        [
            {"id": 1, "passage": "A short passage with only a few words."},
            {"id": 2, "passage": "Another short passage."},
        ],
    )
    corpus = load_corpus(split="train", data_dir=data_dir)
    assert len(corpus) == 2
    ids = {item["id"] for item in corpus}
    assert ids == {1, 2}
    for item in corpus:
        assert item["parent_id"] == item["id"]
        assert "passage" in item


def test_load_corpus_splits_long_passage_into_chunks(tmp_path):
    long_text = " ".join(f"word{i}" for i in range(50)) + ". " + \
        " ".join(f"more{i}" for i in range(50)) + "."
    # Build a passage well above a small threshold so it's guaranteed to split.
    sentences = [f"Sentence number {i} has several words in it." for i in range(30)]
    long_passage = " ".join(sentences)
    data_dir = _write_corpus(
        tmp_path,
        [{"id": 42, "passage": long_passage}],
    )
    corpus = load_corpus(split="train", data_dir=data_dir, threshold=50)
    assert len(corpus) > 1
    for item in corpus:
        assert item["parent_id"] == 42
        assert parent_id_of(item["id"]) == 42
        assert isinstance(item["id"], str)
        assert "::" in item["id"]


def test_load_corpus_default_threshold_leaves_short_passages_as_ids(tmp_path):
    # With the default threshold=500, a passage well under that word count
    # must come back with chunk id == parent id == original passage id, so
    # BM25/dense indexing and eval scoring are unaffected (Phase 1 parity).
    data_dir = _write_corpus(
        tmp_path,
        [{"id": 777, "passage": "Short passage well under five hundred words."}],
    )
    corpus = load_corpus(split="train", data_dir=data_dir)
    assert corpus == [
        {
            "id": 777,
            "passage": "Short passage well under five hundred words.",
            "parent_id": 777,
        }
    ]
