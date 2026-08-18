import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np
from openai import OpenAI

from src.data.loaders import load_corpus

EMBEDDING_MODEL = "text-embedding-3-small"


def _get_embedding_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "Set OPENAI_API_KEY in setup/.env to use hybrid retrieval (dense embeddings)"
        )
    return OpenAI(api_key=api_key)


def embed_texts(
    texts: List[str],
    client: Optional[OpenAI] = None,
    batch_size: int = 100,
) -> np.ndarray:
    client = client or _get_embedding_client()
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        vectors.extend(item.embedding for item in resp.data)
    arr = np.array(vectors, dtype="float32")
    faiss.normalize_L2(arr)
    return arr


def build_dense_index(
    corpus: List[dict] = None,
    client: Optional[OpenAI] = None,
) -> Tuple["faiss.Index", List[dict], dict]:
    if corpus is None:
        corpus = load_corpus()
    from src.data.lookup import build_lookup

    texts = [item["passage"] for item in corpus]
    vectors = embed_texts(texts, client=client)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    id_to_passage = build_lookup(corpus)
    return index, corpus, id_to_passage


def dense_search(
    query: str,
    index: "faiss.Index",
    corpus: List[dict],
    id_to_passage: dict,
    k: int = 5,
    client: Optional[OpenAI] = None,
) -> List[Tuple[int, str, float]]:
    q_vec = embed_texts([query], client=client)
    scores, indices = index.search(q_vec, k)
    out = []
    for idx, score in zip(indices[0], scores[0]):
        if idx == -1:
            continue
        i = int(idx)
        pid = corpus[i]["id"]
        text = id_to_passage[pid]
        out.append((int(pid), text, float(score)))
    return out


def save_index(index: "faiss.Index", corpus: List[dict], path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path / "index.faiss"))
    with open(path / "corpus_ids.json", "w") as f:
        json.dump([item["id"] for item in corpus], f)


def load_index(path: Path, corpus: List[dict]) -> "faiss.Index":
    index_path = path / "index.faiss"
    if not index_path.exists():
        raise FileNotFoundError(
            f'No dense index found at {path}. Run scripts/build_embeddings.py first, '
            f'or set mode="bm25" in config.toml.'
        )
    with open(path / "corpus_ids.json") as f:
        saved_ids = json.load(f)
    current_ids = [item["id"] for item in corpus]
    if saved_ids != current_ids:
        raise ValueError(
            "Dense index corpus order does not match the current corpus. "
            "Re-run scripts/build_embeddings.py to rebuild it."
        )
    return faiss.read_index(str(index_path))
