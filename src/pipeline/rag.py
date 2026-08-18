import logging
from pathlib import Path
from typing import List, Tuple, Any, Dict, Optional

try:
    import tomllib
except ImportError:
    import tomli as tomllib

import httpx

from src.retrieval.sparse import build_index, search as bm25_search
from src.retrieval.dense import load_index as load_dense_index, dense_search
from src.retrieval.fusion import rrf_fuse
from src.retrieval.rerank import rerank as voyage_rerank
from src.prompts.formatter import format_passages_as_context
from src.prompts.templates import build_rag_prompt
from src.generation.llm import generate
from src.generation.parser import parse_answer

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _load_config(path: Path = None) -> Dict[str, Any]:
    path = path or (_project_root() / "config.toml")
    defaults = {
        "retrieval": {
            "mode": "bm25",
            "top_k": 5,
            "sparse_top_n": 20,
            "dense_top_n": 20,
            "fusion_k": 60,
            "rerank": False,
            "rerank_top_n": 5,
        },
        "llm": {"model": "llama-3.3-70b-versatile", "max_tokens": 512, "temperature": 0.2},
    }
    if not path.exists():
        return defaults
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return {
        "retrieval": {**defaults["retrieval"], **(data.get("retrieval") or {})},
        "llm": {**defaults["llm"], **(data.get("llm") or {})},
    }


# Retrieval indices, loaded once (see load_indices) and shared across requests.
_bm25_state: Optional[Tuple[Any, List[dict], dict]] = None
_dense_state: Optional[Tuple[Any, List[dict], dict]] = None


def load_indices(mode: str = "bm25") -> None:
    """Build/load retrieval indices. Call once at process startup
    (see scripts/run_server.py) to avoid concurrent lazy-build races
    under FastAPI's threaded request handling."""
    global _bm25_state, _dense_state
    if _bm25_state is None:
        _bm25_state = build_index()
    if mode == "hybrid":
        _, corpus, id_to_passage = _bm25_state
        dense_index = load_dense_index(_project_root() / "artifacts" / "dense_index", corpus)
        _dense_state = (dense_index, corpus, id_to_passage)
    else:
        _dense_state = None


def _ensure_loaded(mode: str) -> None:
    if _bm25_state is None or (mode == "hybrid" and _dense_state is None):
        load_indices(mode)


def retrieve_candidates(
    question: str,
    r_cfg: Dict[str, Any],
) -> List[Tuple[int, str, float]]:
    """Run BM25 (+ dense + fusion + optional rerank per r_cfg) and return the
    final top_k candidates. Shared by rag_query and scripts/evaluate_retrieval.py
    so both use the exact same retrieval logic."""
    mode = r_cfg.get("mode", "bm25")
    do_rerank = r_cfg.get("rerank", False)
    top_k = r_cfg.get("top_k", 5)
    sparse_top_n = r_cfg.get("sparse_top_n", 20)
    dense_top_n = r_cfg.get("dense_top_n", 20)

    _ensure_loaded(mode)
    bm25, corpus, id_to_passage = _bm25_state

    fetch_k = sparse_top_n if (mode == "hybrid" or do_rerank) else top_k
    sparse_results = bm25_search(question, bm25, corpus, id_to_passage, k=fetch_k)

    if mode == "hybrid" and _dense_state is not None:
        dense_index, d_corpus, d_id_to_passage = _dense_state
        dense_results = dense_search(
            question, dense_index, d_corpus, d_id_to_passage, k=dense_top_n
        )
        candidates = rrf_fuse(sparse_results, dense_results, k=r_cfg.get("fusion_k", 60))
    else:
        candidates = sparse_results

    if do_rerank:
        rerank_top_n = max(top_k, r_cfg.get("rerank_top_n", top_k))
        try:
            reranked = voyage_rerank(question, candidates, top_n=rerank_top_n)
            return reranked[:top_k]
        except httpx.HTTPError as e:
            logger.warning("Voyage rerank failed (%s); falling back to unreranked candidates", e)
            return candidates[:top_k]

    return candidates[:top_k]


def rag_query(
    question: str,
    top_k: int = None,
    max_tokens: int = None,
    config: Dict[str, Any] = None,
) -> Tuple[List[Tuple[int, str, float]], str]:
    cfg = config or _load_config()
    r_cfg = dict(cfg.get("retrieval", {}))
    if top_k is not None:
        r_cfg["top_k"] = top_k
    llm_cfg = cfg.get("llm", {})
    max_tokens = max_tokens or llm_cfg.get("max_tokens", 512)
    temperature = llm_cfg.get("temperature", 0.2)
    model = llm_cfg.get("model", "llama-3.3-70b-versatile")

    passages = retrieve_candidates(question, r_cfg)

    context = format_passages_as_context(passages)
    prompt = build_rag_prompt(context=context, question=question)
    raw = generate(prompt, model=model, max_tokens=max_tokens, temperature=temperature)
    answer = parse_answer(raw)
    return passages, answer
