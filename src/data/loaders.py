import json
from pathlib import Path
from typing import List, Dict, Any

from src.data.chunking import chunk_passages


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def load_corpus(split: str = "train", data_dir: Path = None, threshold: int = 500) -> List[Dict[str, Any]]:
    """Load corpus passages and apply chunking.

    Args:
        split: Dataset split (train/test/validation)
        data_dir: Directory containing the corpus JSON files
        threshold: Word count threshold for chunking. Passages with word count <= threshold
                  are kept as-is; longer passages are split at sentence boundaries.
                  Default 500 preserves Phase 1 behavior (most real passages don't split).

    Returns:
        List of dicts with keys: id, passage, parent_id
        - Unchunked passages: id == parent_id
        - Chunked passages: id is "parent_id::chunk_idx", parent_id is original passage id
    """
    if data_dir is None:
        data_dir = _project_root() / "data" / "json" / "text-corpus"
    path = data_dir / f"{split}-00000-of-00001.json"
    with open(path, "r") as f:
        passages = json.load(f)

    # Normalize passage structure: convert "passage" key to "text" for chunking,
    # then chunk, then restore "passage" key for backward compatibility
    normalized = []
    for p in passages:
        normalized.append({
            "id": p["id"],
            "text": p.get("passage", p.get("text", "")),
        })

    # Apply chunking
    chunks = chunk_passages(normalized, threshold=threshold)

    # Restore "passage" key and keep parent_id for eval tracking
    result = []
    for chunk in chunks:
        result.append({
            "id": chunk["id"],
            "passage": chunk["text"],
            "parent_id": chunk["parent_id"],
        })

    return result


def load_qa(split: str = "train", data_dir: Path = None) -> List[Dict[str, Any]]:
    if data_dir is None:
        data_dir = _project_root() / "data" / "json" / "question-answer-passages"
    path = data_dir / f"{split}-00000-of-00001.json"
    with open(path, "r") as f:
        return json.load(f)
