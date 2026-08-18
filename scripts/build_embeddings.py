#!/usr/bin/env python3
"""Build and persist the dense (embedding) index into artifacts/dense_index/.

Run this once after the corpus changes, or before enabling
`mode = "hybrid"` in config.toml for the first time.
"""
import hashlib
import sys
from pathlib import Path
from typing import List

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.loaders import load_corpus
from src.retrieval.dense import build_dense_index, save_index

ARTIFACTS_DIR = project_root / "artifacts" / "dense_index"
CORPUS_PATH = project_root / "data" / "json" / "text-corpus" / "train-00000-of-00001.json"


def corpus_file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_and_persist(
    corpus: List[dict],
    artifacts_dir: Path,
    corpus_hash_value: str,
    client=None,
) -> bool:
    """Build and persist the dense index. Returns True if rebuilt, False if
    the cached index already matches corpus_hash_value."""
    hash_file = artifacts_dir / "corpus.sha256"
    if hash_file.exists() and hash_file.read_text().strip() == corpus_hash_value:
        return False
    index, built_corpus, _ = build_dense_index(corpus, client=client)
    save_index(index, built_corpus, artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    hash_file.write_text(corpus_hash_value)
    return True


def main():
    corpus = load_corpus()
    new_hash = corpus_file_hash(CORPUS_PATH)
    print(f"Embedding {len(corpus)} passages (skipped if unchanged since last run)...")
    built = build_and_persist(corpus, ARTIFACTS_DIR, new_hash)
    if built:
        print(f"Built and saved dense index to {ARTIFACTS_DIR}")
    else:
        print("Dense index already up to date — nothing to do.")


if __name__ == "__main__":
    main()
