import math
from typing import List, Optional, Sequence, Union

Id = Union[int, str]


def _top_k(retrieved_ids: Sequence[Id], k: Optional[int]) -> List[Id]:
    """Truncate a ranked list to k, which the *_at_k functions must do themselves.

    Callers used to be trusted to pass exactly k ids; passing more silently
    produced recall@len(retrieved) under an @k name.
    """
    return list(retrieved_ids) if k is None else list(retrieved_ids)[:k]


def recall_at_k(
    retrieved_ids: Sequence[Id], relevant_ids: Sequence[Id], k: Optional[int] = None
) -> float:
    """Fraction of all gold passages that appear in the top k.

    This is the standard definition, which means the score is capped at
    k / len(relevant_ids) whenever a question has more gold passages than k.
    Use max_recall_at_k to report that ceiling alongside the score, otherwise
    the number reads as far worse than the retriever actually is.
    """
    if not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    retrieved_set = set(_top_k(retrieved_ids, k))
    return len(relevant_set & retrieved_set) / len(relevant_set)


def max_recall_at_k(relevant_ids: Sequence[Id], k: int) -> float:
    """Highest recall_at_k a perfect retriever could score on this question."""
    if not relevant_ids:
        return 0.0
    return min(k, len(set(relevant_ids))) / len(set(relevant_ids))


def mrr_at_k(
    retrieved_ids: Sequence[Id], relevant_ids: Sequence[Id], k: Optional[int] = None
) -> float:
    relevant_set = set(relevant_ids)
    for rank, rid in enumerate(_top_k(retrieved_ids, k), start=1):
        if rid in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_ids: Sequence[Id], relevant_ids: Sequence[Id], k: Optional[int] = None
) -> float:
    relevant_set = set(relevant_ids)
    top = _top_k(retrieved_ids, k)
    dcg = 0.0
    for rank, rid in enumerate(top, start=1):
        if rid in relevant_set:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant_set), len(top))
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0
