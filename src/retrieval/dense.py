import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import httpx
import numpy as np

from src.config.env import load_env
from src.data.loaders import load_corpus

load_env()

VOYAGE_EMBEDDINGS_URL = "https://api.voyageai.com/v1/embeddings"
EMBEDDING_MODEL = "voyage-3-lite"


def _get_api_key() -> str:
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        raise ValueError(
            "Set VOYAGE_API_KEY in setup/.env to use hybrid retrieval (dense embeddings)"
        )
    return api_key


def embed_texts(
    texts: List[str],
    client: Optional[httpx.Client] = None,
    batch_size: int = 100,
) -> np.ndarray:
    api_key = _get_api_key()
    headers = {"Authorization": f"Bearer {api_key}"}
    owns_client = client is None
    client = client or httpx.Client(timeout=30.0)
    vectors = []
    try:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            payload = {
                "input": batch,
                "model": EMBEDDING_MODEL,
                "input_type": "document",
            }
            resp = client.post(VOYAGE_EMBEDDINGS_URL, json=payload, headers=headers)
            resp.raise_for_status()
            vectors.extend(item["embedding"] for item in resp.json()["data"])
    finally:
        if owns_client:
            client.close()
    arr = np.array(vectors, dtype="float32")
    faiss.normalize_L2(arr)
    return arr


def build_dense_index(
    corpus: List[dict] = None,
    client: Optional[httpx.Client] = None,
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
    client: Optional[httpx.Client] = None,
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
