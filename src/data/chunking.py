import re
from typing import Any, Dict, List, Union

# Sentence boundary: punctuation followed by whitespace. Uses a lookbehind so
# the punctuation stays attached to the sentence it ends (splitting *between*
# sentences rather than consuming the boundary characters).
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

_DEFAULT_OVERLAP_WORDS = 50


def _effective_overlap(threshold: int) -> int:
    """Overlap capped at half the chunk budget.

    A fixed 50-word overlap is larger than the whole budget for small
    thresholds, which drove the word-splitter's step size to 1 and made it emit
    one chunk per word. Capping at threshold // 2 keeps every step >= half a
    chunk, so the number of chunks stays linear in the input.
    """
    return max(1, min(_DEFAULT_OVERLAP_WORDS, threshold // 2))


def chunk_passages(corpus: List[Dict[str, Any]], threshold: int = 500) -> List[Dict[str, Any]]:
    """Split long passages into overlapping chunks; leave short ones as-is.

    Each passage in `corpus` must have `id` and `text` keys. Passages whose
    word count (whitespace-split) is <= threshold are returned unchanged,
    with chunk id equal to the passage id. Passages above threshold are
    split at sentence boundaries into chunks of ~threshold words each, with
    ~50 words of overlap carried from the end of one chunk to the start of
    the next; chunk ids are formatted as "{parent_id}::{chunk_idx}".
    """
    chunks: List[Dict[str, Any]] = []
    for passage in corpus:
        parent_id = passage["id"]
        text = passage["text"]
        word_count = len(text.split())

        if word_count <= threshold:
            chunks.append({"id": parent_id, "text": text, "parent_id": parent_id})
        else:
            chunks.extend(_split_long_passage(parent_id, text, threshold))

    return chunks


def parent_id_of(chunk_id: Union[str, int]) -> int:
    """Extract the parent passage id from a chunk id.

    Accepts either a chunked id ("parent_id::chunk_idx") or a bare
    non-chunked id ("parent_id" or an int), and always returns the parent
    id as an int.
    """
    parent_part = str(chunk_id).split("::", 1)[0]
    return int(parent_part)


def _split_long_passage(parent_id: Any, text: str, threshold: int) -> List[Dict[str, Any]]:
    sentences = _split_into_sentences(text)

    if len(sentences) <= 1:
        # No usable sentence boundaries (or only one sentence) — fall back
        # to splitting on raw word count so the passage still gets chunked.
        return _split_by_words(parent_id, text.split(), threshold)

    return _group_sentences_into_chunks(parent_id, sentences, threshold)


def _split_into_sentences(text: str) -> List[str]:
    sentences = _SENTENCE_BOUNDARY.split(text.strip())
    return [s for s in sentences if s]


def _group_sentences_into_chunks(
    parent_id: Any, sentences: List[str], threshold: int
) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    current_sentences: List[str] = []
    current_word_count = 0
    chunk_idx = 0

    def flush() -> None:
        nonlocal chunk_idx, current_sentences, current_word_count
        chunks.append(_make_chunk(parent_id, chunk_idx, " ".join(current_sentences)))
        chunk_idx += 1
        current_sentences = _overlap_tail(current_sentences, _effective_overlap(threshold))
        current_word_count = sum(len(s.split()) for s in current_sentences)

    for sentence in sentences:
        sentence_word_count = len(sentence.split())

        if current_sentences and current_word_count + sentence_word_count > threshold:
            flush()

        if sentence_word_count > threshold:
            # A single sentence longer than the whole budget cannot be grouped;
            # carrying it whole would produce a chunk far over threshold and, as
            # overlap, duplicate it into the next chunk too. Split it by words.
            if current_sentences:
                flush()
            for part in _split_by_words(parent_id, sentence.split(), threshold):
                part["id"] = f"{parent_id}::{chunk_idx}"
                chunk_idx += 1
                chunks.append(part)
            current_sentences = []
            current_word_count = 0
            continue

        current_sentences.append(sentence)
        current_word_count += sentence_word_count

    if current_sentences:
        chunks.append(_make_chunk(parent_id, chunk_idx, " ".join(current_sentences)))

    return chunks


def _overlap_tail(sentences: List[str], target_words: int) -> List[str]:
    """Return the trailing sentences whose combined word count is >= target_words.

    Never returns every sentence: carrying the whole chunk forward as overlap
    would duplicate it verbatim into the next chunk and make chunks grow without
    bound, so at least the first sentence is always dropped.
    """
    if len(sentences) <= 1:
        return []

    tail: List[str] = []
    word_count = 0
    for sentence in reversed(sentences[1:]):
        tail.insert(0, sentence)
        word_count += len(sentence.split())
        if word_count >= target_words:
            break
    return tail


def _split_by_words(parent_id: Any, words: List[str], threshold: int) -> List[Dict[str, Any]]:
    """Fallback chunking by raw word count when no sentence boundaries exist."""
    chunks: List[Dict[str, Any]] = []
    chunk_idx = 0
    start = 0

    while start < len(words):
        end = start + threshold
        chunk_words = words[start:end]
        chunks.append(_make_chunk(parent_id, chunk_idx, " ".join(chunk_words)))
        chunk_idx += 1

        if end >= len(words):
            break

        start += threshold - _effective_overlap(threshold)

    return chunks


def _make_chunk(parent_id: Any, chunk_idx: int, text: str) -> Dict[str, Any]:
    return {"id": f"{parent_id}::{chunk_idx}", "text": text, "parent_id": parent_id}
