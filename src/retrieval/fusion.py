from typing import Dict, List, Optional, Tuple


def rrf_fuse(
    sparse_results: List[Tuple[int, str, float]],
    dense_results: List[Tuple[int, str, float]],
    k: int = 60,
    top_n: Optional[int] = None,
) -> List[Tuple[int, str, float]]:
    scores: Dict[Union[int, str], float] = {}
    texts: Dict[Union[int, str], str] = {}
    for rank, (pid, text, _) in enumerate(sparse_results, start=1):
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
        texts[pid] = text
    for rank, (pid, text, _) in enumerate(dense_results, start=1):
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
        texts[pid] = text
    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if top_n is not None:
        fused = fused[:top_n]
    return [(pid, texts[pid], score) for pid, score in fused]
