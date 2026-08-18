import math
from typing import List


def recall_at_k(retrieved_ids: List[int], relevant_ids: List[int]) -> float:
    if not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    retrieved_set = set(retrieved_ids)
    return len(relevant_set & retrieved_set) / len(relevant_set)


def mrr_at_k(retrieved_ids: List[int], relevant_ids: List[int]) -> float:
    relevant_set = set(relevant_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: List[int], relevant_ids: List[int]) -> float:
    relevant_set = set(relevant_ids)
    dcg = 0.0
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_set:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant_set), len(retrieved_ids))
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0
