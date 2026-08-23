from src.data.chunking import chunk_passages, parent_id_of


def _make_sentence(index: int, word_count: int) -> str:
    """Build a deterministic sentence with exactly `word_count` words."""
    words = [f"s{index}w{i}" for i in range(word_count)]
    return " ".join(words) + "."


def test_passage_below_threshold_kept_as_is():
    corpus = [{"id": 1, "text": "A short passage with only a few words."}]
    chunks = chunk_passages(corpus, threshold=500)
    assert len(chunks) == 1
    assert chunks[0]["id"] == 1
    assert chunks[0]["parent_id"] == 1
    assert chunks[0]["text"] == "A short passage with only a few words."


def test_passage_above_threshold_is_split():
    sentences = [_make_sentence(i, 20) for i in range(10)]
    text = " ".join(sentences)
    corpus = [{"id": 42, "text": text}]
    chunks = chunk_passages(corpus, threshold=100)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk["parent_id"] == 42


def test_chunk_ids_formatted_as_parent_id_chunk_idx():
    sentences = [_make_sentence(i, 20) for i in range(10)]
    text = " ".join(sentences)
    corpus = [{"id": 42, "text": text}]
    chunks = chunk_passages(corpus, threshold=100)
    for idx, chunk in enumerate(chunks):
        assert chunk["id"] == f"42::{idx}"


def test_chunk_id_of_non_chunked_passage_equals_passage_id():
    corpus = [{"id": 7, "text": "Short passage."}]
    chunks = chunk_passages(corpus, threshold=500)
    assert chunks[0]["id"] == 7


def test_parent_id_of_extracts_parent_from_chunked_id():
    assert parent_id_of("12345::0") == 12345
    assert parent_id_of("12345::7") == 12345


def test_parent_id_of_handles_non_chunked_id():
    assert parent_id_of("12345") == 12345
    assert parent_id_of(12345) == 12345


def test_no_sentence_boundaries_falls_back_to_word_split():
    # No '.', '!' or '?' anywhere in the text.
    words = [f"word{i}" for i in range(300)]
    text = " ".join(words)
    corpus = [{"id": 99, "text": text}]
    chunks = chunk_passages(corpus, threshold=100)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk["parent_id"] == 99
        assert parent_id_of(chunk["id"]) == 99
    # No content should be lost or garbled: rejoining unique words covers input.
    all_words = set()
    for chunk in chunks:
        all_words.update(chunk["text"].split())
    assert all_words == set(words)


def test_all_passages_below_threshold():
    corpus = [
        {"id": 1, "text": "First short passage."},
        {"id": 2, "text": "Second short passage."},
        {"id": 3, "text": "Third short passage."},
    ]
    chunks = chunk_passages(corpus, threshold=500)
    assert len(chunks) == 3
    ids = {c["id"] for c in chunks}
    assert ids == {1, 2, 3}
    for c in chunks:
        assert c["id"] == c["parent_id"]


def test_all_passages_above_threshold():
    corpus = []
    for pid in (1, 2, 3):
        sentences = [_make_sentence(i, 20) for i in range(10)]
        corpus.append({"id": pid, "text": " ".join(sentences)})
    chunks = chunk_passages(corpus, threshold=100)
    parent_ids_seen = {c["parent_id"] for c in chunks}
    assert parent_ids_seen == {1, 2, 3}
    assert len(chunks) > 3  # each passage split into multiple chunks


def test_empty_corpus_returns_empty_list():
    assert chunk_passages([], threshold=500) == []


def test_overlap_between_consecutive_chunks():
    # 10 sentences x 20 words = 200 words, threshold=100 forces a split.
    sentences = [_make_sentence(i, 20) for i in range(10)]
    text = " ".join(sentences)
    corpus = [{"id": 5, "text": text}]
    chunks = chunk_passages(corpus, threshold=100)
    assert len(chunks) >= 2

    first_words = chunks[0]["text"].split()
    second_words = chunks[1]["text"].split()

    # The tail of chunk 0 should reappear at the head of chunk 1 (overlap),
    # and the overlap should be roughly 50 words (allowing for sentence granularity).
    overlap_words = []
    for w in first_words[::-1]:
        if second_words[: len(overlap_words) + 1] == first_words[len(first_words) - len(overlap_words) - 1 :]:
            overlap_words.append(w)
        else:
            break

    max_possible_overlap = min(len(first_words), len(second_words))
    found_overlap = 0
    for size in range(max_possible_overlap, 0, -1):
        if first_words[-size:] == second_words[:size]:
            found_overlap = size
            break

    assert found_overlap >= 20  # at least one overlapping sentence carried over
    assert found_overlap <= 80  # not the entire chunk duplicated


def test_no_infinite_loop_when_threshold_below_overlap():
    # Regression test for threshold=30, no sentence boundaries
    corpus = [{"id": 1, "text": " ".join(f"word{i}" for i in range(120))}]
    # Should complete without hanging
    chunks = chunk_passages(corpus, threshold=30)
    assert len(chunks) > 0  # At least one chunk
    assert all(len(c["text"].split()) > 0 for c in chunks)  # No empty chunks


def test_word_count_measured_on_whitespace_split():
    # Exactly at threshold should be kept as-is (<=), not split.
    text = " ".join(f"w{i}" for i in range(500)) + "."
    corpus = [{"id": 1, "text": text}]
    chunks = chunk_passages(corpus, threshold=500)
    assert len(chunks) == 1
    assert chunks[0]["id"] == 1
