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


def _retrieve_fn_from_config(r_cfg):
    from src.pipeline.rag import retrieve_candidates

    def _fn(question):
        return [pid for pid, _, _ in retrieve_candidates(question, r_cfg)]

    return _fn


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality against gold relevant_passage_ids.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N QA pairs (cost control)")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    qa_pairs = load_qa(split="test")
    if args.limit:
        qa_pairs = qa_pairs[: args.limit]

    from src.pipeline.rag import _load_config, load_indices as load_pipeline_indices

    cfg = _load_config()
    base_r_cfg = dict(cfg.get("retrieval", {}))
    if args.top_k:
        base_r_cfg["top_k"] = args.top_k

    dense_index_path = project_root / "artifacts" / "dense_index"
    has_dense = (dense_index_path / "index.faiss").exists()

    load_pipeline_indices("hybrid" if has_dense else "bm25")

    results = {}
    bm25_cfg = {**base_r_cfg, "mode": "bm25", "rerank": False}
    results["bm25"] = evaluate_mode(qa_pairs, _retrieve_fn_from_config(bm25_cfg))

    if has_dense:
        hybrid_cfg = {**base_r_cfg, "mode": "hybrid", "rerank": False}
        results["hybrid"] = evaluate_mode(qa_pairs, _retrieve_fn_from_config(hybrid_cfg))

        hybrid_rerank_cfg = {**base_r_cfg, "mode": "hybrid", "rerank": True}
        results["hybrid_rerank"] = evaluate_mode(qa_pairs, _retrieve_fn_from_config(hybrid_rerank_cfg))
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
