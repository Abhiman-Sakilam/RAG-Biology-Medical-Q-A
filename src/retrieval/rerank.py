import logging
import os
from typing import List, Optional, Tuple, Union

import httpx

from src.config.env import load_env

load_env()

logger = logging.getLogger(__name__)

# Chat endpoint: the request sends `messages` and the reply is read from
# choices[0].message.content, both of which are the chat contract. The
# text-completions endpoint takes `prompt` and returns choices[0].text.
OPENROUTER_RERANK_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "nvidia/llama-nemotron-rerank-vl-1b-v2:free"


def _get_api_key() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "Set OPENROUTER_API_KEY in setup/.env to use reranking ([retrieval] rerank = true)"
        )
    return api_key


def _extract_content(body: dict) -> str:
    """Pull the model's text out of a chat-completions body.

    A 200 response with an unexpected shape must not raise: the caller treats a
    reranking failure as "keep the fusion order", which is a far better outcome
    than a failed query.
    """
    try:
        message = body["choices"][0]["message"]
        return message.get("content") or ""
    except (KeyError, IndexError, TypeError) as e:
        logger.warning("Unexpected rerank response shape (%s); ignoring rerank", e)
        return ""


def _parse_ranked_indices(result_text: str, num_candidates: int) -> List[int]:
    """Extract an ordered, de-duplicated list of valid candidate indices from
    the model's free-text response.

    The model is asked to return "one index per line", but is not guaranteed
    to comply exactly (it may prefix with "Index: ", add surrounding text,
    etc.), so each line is scanned for its leading integer rather than
    requiring the line to be a bare number.
    """
    ranked_indices: List[int] = []
    seen = set()
    for line in result_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        digits = ""
        for ch in line:
            if ch.isdigit() or (ch == "-" and not digits):
                digits += ch
            elif digits:
                break
        if not digits or digits == "-":
            continue
        idx = int(digits)
        if 0 <= idx < num_candidates and idx not in seen:
            ranked_indices.append(idx)
            seen.add(idx)
    return ranked_indices


def rerank(
    query: str,
    candidates: List[Tuple[Union[int, str], str, float]],
    top_n: int = 5,
    api_key: Optional[str] = None,
    client: Optional[httpx.Client] = None,
) -> List[Tuple[Union[int, str], str, float]]:
    if not candidates:
        return []

    api_key = api_key or _get_api_key()

    docs_text = "\n".join(
        f"{i}. {text[:200]}" for i, (_, text, _) in enumerate(candidates)
    )
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Rank these documents by relevance to the query:\n\n"
                    f"Query: {query}\n\nDocuments:\n{docs_text}\n\n"
                    f"Return the top {top_n} document indices in order of "
                    "relevance, one per line."
                ),
            }
        ],
        "temperature": 0,
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    owns_client = client is None
    client = client or httpx.Client(timeout=30.0)
    try:
        resp = client.post(OPENROUTER_RERANK_URL, json=payload, headers=headers)
        resp.raise_for_status()
        result_text = _extract_content(resp.json())
    finally:
        if owns_client:
            client.close()

    ranked_indices = _parse_ranked_indices(result_text, len(candidates))[:top_n]
    if not ranked_indices:
        # The model's response didn't yield any parseable indices; fall back
        # to the original (fusion-ranked) candidate order rather than
        # dropping the reranking step's results entirely.
        return candidates[:top_n]

    out = []
    n = len(ranked_indices)
    for rank, idx in enumerate(ranked_indices):
        pid, text, _ = candidates[idx]
        score = (n - rank) / n
        out.append((pid, text, score))
    return out
