import os
from typing import List, Optional, Tuple

import httpx

VOYAGE_RERANK_URL = "https://api.voyageai.com/v1/rerank"
VOYAGE_MODEL = "rerank-2"


def _get_api_key() -> str:
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        raise ValueError(
            "Set VOYAGE_API_KEY in setup/.env to use reranking ([retrieval] rerank = true)"
        )
    return api_key


def rerank(
    query: str,
    candidates: List[Tuple[int, str, float]],
    top_n: int = 5,
    api_key: Optional[str] = None,
    client: Optional[httpx.Client] = None,
) -> List[Tuple[int, str, float]]:
    if not candidates:
        return []
    api_key = api_key or _get_api_key()
    documents = [text for _, text, _ in candidates]
    payload = {
        "query": query,
        "documents": documents,
        "model": VOYAGE_MODEL,
        "top_k": min(top_n, len(candidates)),
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    owns_client = client is None
    client = client or httpx.Client(timeout=10.0)
    try:
        resp = client.post(VOYAGE_RERANK_URL, json=payload, headers=headers)
        resp.raise_for_status()
        results = resp.json()["results"]
    finally:
        if owns_client:
            client.close()
    out = []
    for item in results:
        pid, text, _ = candidates[item["index"]]
        out.append((pid, text, float(item["relevance_score"])))
    return out
