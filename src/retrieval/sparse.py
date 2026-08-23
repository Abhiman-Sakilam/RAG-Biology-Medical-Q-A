import re
from typing import List, Tuple, Union

from rank_bm25 import BM25Okapi

from src.data.loaders import load_corpus
from src.data.lookup import build_lookup


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def build_index(corpus: List[dict] = None) -> Tuple[BM25Okapi, List[dict], dict]:
    """Build BM25 index over corpus (handles both passages and chunks).

    Args:
        corpus: List of passage/chunk dicts. If None, loads corpus with chunking.
                Each item should have "id" and "passage"/"text" keys.

    Returns:
        Tuple of (bm25_index, corpus, id_to_text_lookup)
    """
    if corpus is None:
        corpus = load_corpus()

    # Handle both "passage" (original) and "text" (chunked) keys
    tokenized = [_tokenize(item.get("passage", item.get("text", ""))) for item in corpus]
    bm25 = BM25Okapi(tokenized)
    id_to_passage = build_lookup(corpus)
    return bm25, corpus, id_to_passage


def search(
    query: str,
    bm25: BM25Okapi,
    corpus: List[dict],
    id_to_passage: dict,
    k: int = 5,
) -> List[Tuple[Union[int, str], str, float]]:
    """Search BM25 index and return top-k results.

    Returns list of (chunk_id, text, score) tuples.
    chunk_id may be an int (non-chunked passage) or string (chunked id like "parent::0").
    """
    q_tokens = _tokenize(query)
    scores = bm25.get_scores(q_tokens)
    top_indices = scores.argsort()[::-1][:k]
    out = []
    for idx in top_indices:
        i = int(idx)
        pid = corpus[i]["id"]
        text = id_to_passage[pid]
        out.append((pid, text, float(scores[i])))
    return out
