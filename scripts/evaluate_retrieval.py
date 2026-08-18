#!/usr/bin/env python3
"""Evaluate retrieval quality (Recall/MRR/nDCG) for bm25 vs hybrid vs
hybrid+rerank against the gold relevant_passage_ids in the test QA split.

This is the Phase 1 acceptance gate: hybrid+rerank should show a
measurable improvement over the bm25 baseline before Phase 1 is
considered done.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.loaders import load_qa
from src.eval.metrics import recall_at_k, mrr_at_k, ndcg_at_k


def evaluate_mode(
    qa_pairs: List[dict],
    retrieve_fn: Callable[[str], List[int]],
) -> Dict[str, float]:
    recalls, mrrs, ndcgs = [], [], []
    for qa in qa_pairs:
        retrieved_ids = retrieve_fn(qa["question"])
        gold_ids = qa["relevant_passage_ids"]
        recalls.append(recall_at_k(retrieved_ids, gold_ids))
        mrrs.append(mrr_at_k(retrieved_ids, gold_ids))
        ndcgs.append(ndcg_at_k(retrieved_ids, gold_ids))
    n = len(qa_pairs)
    return {
        "recall_at_k": sum(recalls) / n,
        "mrr_at_k": sum(mrrs) / n,
        "ndcg_at_k": sum(ndcgs) / n,
    }


def _bm25_retrieve_fn(bm25, corpus, id_to_passage, k):
    from src.retrieval.sparse import search

    def _fn(question):
        return [pid for pid, _, _ in search(question, bm25, corpus, id_to_passage, k=k)]

    return _fn


def _hybrid_retrieve_fn(
    bm25, corpus, id_to_passage, dense_index, k, sparse_top_n, dense_top_n, fusion_k, rerank_top_n=None
):
    from src.retrieval.sparse import search as bm25_search
    from src.retrieval.dense import dense_search
    from src.retrieval.fusion import rrf_fuse
    from src.retrieval.rerank import rerank as voyage_rerank

    def _fn(question):
        sparse = bm25_search(question, bm25, corpus, id_to_passage, k=sparse_top_n)
        dense = dense_search(question, dense_index, corpus, id_to_passage, k=dense_top_n)
        fused = rrf_fuse(sparse, dense, k=fusion_k)
        if rerank_top_n:
            candidate_pool = fused[: max(sparse_top_n, dense_top_n)]
            reranked = voyage_rerank(question, candidate_pool, top_n=rerank_top_n)
            return [pid for pid, _, _ in reranked[:k]]
        return [pid for pid, _, _ in fused[:k]]

    return _fn


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality against gold relevant_passage_ids.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N QA pairs (cost control)")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    qa_pairs = load_qa(split="test")
    if args.limit:
        qa_pairs = qa_pairs[: args.limit]

    from src.retrieval.sparse import build_index
    from src.retrieval.dense import load_index as load_dense_index

    bm25, corpus, id_to_passage = build_index()
    results = {"bm25": evaluate_mode(qa_pairs, _bm25_retrieve_fn(bm25, corpus, id_to_passage, k=args.top_k))}

    dense_index_path = project_root / "artifacts" / "dense_index"
    if (dense_index_path / "index.faiss").exists():
        dense_index = load_dense_index(dense_index_path, corpus)
        results["hybrid"] = evaluate_mode(
            qa_pairs,
            _hybrid_retrieve_fn(
                bm25, corpus, id_to_passage, dense_index,
                k=args.top_k, sparse_top_n=20, dense_top_n=20, fusion_k=60,
            ),
        )
        results["hybrid_rerank"] = evaluate_mode(
            qa_pairs,
            _hybrid_retrieve_fn(
                bm25, corpus, id_to_passage, dense_index,
                k=args.top_k, sparse_top_n=20, dense_top_n=20, fusion_k=60, rerank_top_n=args.top_k,
            ),
        )
    else:
        print(
            "No dense index found at artifacts/dense_index — run scripts/build_embeddings.py "
            "first to include hybrid/hybrid_rerank modes.",
            file=sys.stderr,
        )

    print(json.dumps(results, indent=2))

    out_dir = project_root / "artifacts" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "retrieval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote results to {out_path}")


if __name__ == "__main__":
    main()
