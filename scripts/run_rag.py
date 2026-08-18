#!/usr/bin/env python3
"""CLI for RAG: pass a question, get answer and optional passage list."""
import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.rag import rag_query


def main():
    parser = argparse.ArgumentParser(description="RAG: ask a question, get an answer from the corpus.")
    parser.add_argument("--question", "-q", type=str, help="Question to ask")
    parser.add_argument("--out", type=str, help="Write full result (answer + passages) to JSON file")
    parser.add_argument("--top-k", type=int, default=None, help="Number of passages to retrieve")
    args = parser.parse_args()

    question = args.question
    if not question:
        question = sys.stdin.read().strip()
    if not question:
        print("Provide --question or pipe question on stdin.", file=sys.stderr)
        sys.exit(1)

    try:
        passages, answer = rag_query(question, top_k=args.top_k)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    result = {
        "question": question,
        "answer": answer,
        "passages": [{"passage_id": p[0], "passage": p[1], "score": p[2]} for p in passages],
    }

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote result to {args.out}")

    print(answer)
    if not args.out:
        print("\n--- Top passages ---", file=sys.stderr)
        for i, (pid, text, score) in enumerate(passages, 1):
            print(f"[{i}] id={pid} score={score:.4f}: {text[:120]}...", file=sys.stderr)


if __name__ == "__main__":
    main()
