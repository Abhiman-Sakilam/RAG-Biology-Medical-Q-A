import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import faiss
import httpx
import numpy as np

from src.config.env import load_env
from src.data.loaders import load_corpus

load_env()

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
EMBEDDING_MODEL = "liquid/lfm2.5-embedding-350m"

# OpenRouter's free tier enforces a rate limit on this endpoint. We pause once
# accumulated usage crosses a safety margin below that ceiling, and we retry a
# handful of times if we still get rate-limited.
EMBEDDING_TPM_LIMIT = 10_000
EMBEDDING_TPM_SAFETY_MARGIN = 9_000
MAX_RATE_LIMIT_RETRIES = 3
RATE_LIMIT_RETRY_BACKOFF_SECONDS = 5


def _get_embedding_client() -> httpx.Client:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "Set OPENROUTER_API_KEY in setup/.env to use hybrid retrieval (dense embeddings)"
        )
    return httpx.Client(
        base_url=OPENROUTER_BASE_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30.0,
    )


def _post_embeddings_with_retry(client: httpx.Client, payload: dict) -> dict:
    """POST a single embeddings batch, retrying on 429s with a backoff sleep."""
    for attempt in range(MAX_RATE_LIMIT_RETRIES):
        try:
            resp = client.post("/embeddings", json=payload)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            is_rate_limited = e.response.status_code == 429
            if is_rate_limited and attempt < MAX_RATE_LIMIT_RETRIES - 1:
                wait_time = RATE_LIMIT_RETRY_BACKOFF_SECONDS * (attempt + 1)
                logger.warning(
                    "OpenRouter API rate limited (429); retrying in %ds (attempt %d/%d)",
                    wait_time,
                    attempt + 1,
                    MAX_RATE_LIMIT_RETRIES,
                )
                time.sleep(wait_time)
            else:
                raise
    raise RuntimeError("unreachable")  # loop always returns or raises


def embed_texts(
    texts: List[str],
    client: Optional[httpx.Client] = None,
    batch_size: int = 10,
) -> np.ndarray:
    owns_client = client is None
    client = client or _get_embedding_client()
    vectors = []
    tokens_this_minute = 0
    minute_start = time.monotonic()
    try:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            payload = {
                "input": batch,
                "model": EMBEDDING_MODEL,
                "encoding_format": "float",
            }

            elapsed = time.monotonic() - minute_start
            if tokens_this_minute >= EMBEDDING_TPM_SAFETY_MARGIN and elapsed < 60:
                sleep_time = 60 - elapsed
                logger.warning(
                    "Approaching OpenRouter TPM limit (%d tokens used this minute); "
                    "sleeping %.1fs",
                    tokens_this_minute,
                    sleep_time,
                )
                time.sleep(sleep_time)
                tokens_this_minute = 0
                minute_start = time.monotonic()

            data = _post_embeddings_with_retry(client, payload)
            vectors.extend(item["embedding"] for item in data["data"])
            tokens_this_minute += data.get("usage", {}).get("prompt_tokens", 0)

            # Add delay between batches to respect rate limits
            if i + batch_size < len(texts):  # Not the last batch
                time.sleep(2)
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
) -> List[Tuple[Union[int, str], str, float]]:
    """Search dense index and return top-k results.

    Returns list of (chunk_id, text, score) tuples.
    chunk_id may be an int (non-chunked passage) or string (chunked id like "parent::0").
    """
    q_vec = embed_texts([query], client=client)
    scores, indices = index.search(q_vec, k)
    out = []
    for idx, score in zip(indices[0], scores[0]):
        if idx == -1:
            continue
        i = int(idx)
        pid = corpus[i]["id"]
        text = id_to_passage[pid]
        out.append((pid, text, float(score)))
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
