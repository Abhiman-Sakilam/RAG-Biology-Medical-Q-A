#!/usr/bin/env python3
"""Evaluate retrieval quality (Recall/MRR/nDCG) for bm25 vs hybrid vs
hybrid+rerank against the gold relevant_passage_ids in the test QA split.

This is the Phase 1 acceptance gate: hybrid+rerank should show a
measurable improvement over the bm25 baseline before Phase 1 is
considered done.

Task 2: With chunking enabled, this script maps chunk_id -> parent_id before
scoring, ensuring eval metrics match Phase 1 baseline (0.502/0.833/0.735).
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Union

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.loaders import load_qa
from src.data.chunking import parent_id_of
from src.eval.metrics import recall_at_k, mrr_at_k, ndcg_at_k, max_recall_at_k


def evaluate_mode(
    qa_pairs: List[dict],
    retrieve_fn: Callable[[str], List[Union[int, str]]],
    k: int = 5,
) -> Dict[str, float]:
    """Evaluate retrieval mode, mapping chunk_ids to parent_ids before scoring.

    retrieve_fn returns chunk or passage ids (may be strings like "parent::0").
    This function maps them back to parent passage ids before comparing against gold.
    """
    recalls, mrrs, ndcgs, ceilings = [], [], [], []
    for qa in qa_pairs:
        retrieved_ids = retrieve_fn(qa["question"])
        gold_ids = qa["relevant_passage_ids"]

        # Task 2: Map chunk ids back to parent passage ids for scoring
        retrieved_parent_ids = [parent_id_of(cid) for cid in retrieved_ids]

        recalls.append(recall_at_k(retrieved_parent_ids, gold_ids, k=k))
        mrrs.append(mrr_at_k(retrieved_parent_ids, gold_ids, k=k))
        ndcgs.append(ndcg_at_k(retrieved_parent_ids, gold_ids, k=k))
        ceilings.append(max_recall_at_k(gold_ids, k))
    n = len(qa_pairs)
    ceiling = sum(ceilings) / n
    recall = sum(recalls) / n
    return {
        "recall_at_k": recall,
        "mrr_at_k": sum(mrrs) / n,
        "ndcg_at_k": sum(ndcgs) / n,
        # The test set averages 8.66 gold passages per question, so recall@5 can
        # never reach 1.0. Reporting the ceiling makes the score readable.
        "max_recall_at_k": ceiling,
        "recall_vs_ceiling": (recall / ceiling) if ceiling else 0.0,
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
    results["bm25"] = evaluate_mode(qa_pairs, _retrieve_fn_from_config(bm25_cfg), k=args.top_k)

    if has_dense:
        hybrid_cfg = {**base_r_cfg, "mode": "hybrid", "rerank": False}
        results["hybrid"] = evaluate_mode(qa_pairs, _retrieve_fn_from_config(hybrid_cfg), k=args.top_k)

        hybrid_rerank_cfg = {**base_r_cfg, "mode": "hybrid", "rerank": True}
        results["hybrid_rerank"] = evaluate_mode(
        qa_pairs, _retrieve_fn_from_config(hybrid_rerank_cfg), k=args.top_k
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
