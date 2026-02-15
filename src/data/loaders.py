import json
from pathlib import Path
from typing import List, Dict, Any


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def load_corpus(split: str = "train", data_dir: Path = None) -> List[Dict[str, Any]]:
    if data_dir is None:
        data_dir = _project_root() / "data" / "json" / "text-corpus"
    path = data_dir / f"{split}-00000-of-00001.json"
    with open(path, "r") as f:
        return json.load(f)


def load_qa(split: str = "train", data_dir: Path = None) -> List[Dict[str, Any]]:
    if data_dir is None:
        data_dir = _project_root() / "data" / "json" / "question-answer-passages"
    path = data_dir / f"{split}-00000-of-00001.json"
    with open(path, "r") as f:
        return json.load(f)
