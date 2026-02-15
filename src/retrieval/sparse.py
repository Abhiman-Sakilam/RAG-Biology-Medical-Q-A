import re
from typing import List, Tuple

from rank_bm25 import BM25Okapi

from src.data.loaders import load_corpus
from src.data.lookup import build_lookup


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def build_index(corpus: List[dict] = None) -> Tuple[BM25Okapi, List[dict], dict]:
    if corpus is None:
        corpus = load_corpus()
    tokenized = [_tokenize(item["passage"]) for item in corpus]
    bm25 = BM25Okapi(tokenized)
    id_to_passage = build_lookup(corpus)
    return bm25, corpus, id_to_passage


def search(
    query: str,
    bm25: BM25Okapi,
    corpus: List[dict],
    id_to_passage: dict,
    k: int = 5,
) -> List[Tuple[int, str, float]]:
    q_tokens = _tokenize(query)
    scores = bm25.get_scores(q_tokens)
    top_indices = scores.argsort()[::-1][:k]
    out = []
    for idx in top_indices:
        i = int(idx)
        pid = corpus[i]["id"]
        text = id_to_passage[pid]
        out.append((int(pid), text, float(scores[i])))
    return out
