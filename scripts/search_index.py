#!/usr/bin/env python3
"""Search the BM25 index: pass a query and get top-k passages."""
import argparse
import sys
from pathlib import Path

# Allow importing from src when run as script
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.sparse import build_index, search


def main():
    parser = argparse.ArgumentParser(description="Search the passage index with a query.")
    parser.add_argument("query", type=str, help="Search query (question or keywords)")
    parser.add_argument("-k", type=int, default=5, help="Number of passages to return (default: 5)")
    args = parser.parse_args()

    print("Loading corpus and building BM25 index...")
    bm25, corpus, id_to_passage = build_index()
    print(f"Index size: {len(corpus)} passages\n")

    results = search(args.query, bm25, corpus, id_to_passage, k=args.k)
    print(f"Top {len(results)} results for: {args.query!r}\n")
    for i, (pid, text, score) in enumerate(results, 1):
        preview = (text[:200] + "...") if len(text) > 200 else text
        print(f"--- {i} (id={pid}, score={score:.4f}) ---")
        print(preview)
        print()


if __name__ == "__main__":
    main()
