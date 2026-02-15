from pathlib import Path
from typing import List, Tuple, Any, Dict, Optional

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from src.retrieval.sparse import build_index, search
from src.prompts.formatter import format_passages_as_context
from src.prompts.templates import build_rag_prompt
from src.generation.llm import generate
from src.generation.parser import parse_answer


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _load_config() -> Dict[str, Any]:
    path = _project_root() / "config.toml"
    defaults = {"retrieval": {"top_k": 5}, "llm": {"model": "llama-3.3-70b-versatile", "max_tokens": 512, "temperature": 0.2}}
    if not path.exists():
        return defaults
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return {
        "retrieval": {**defaults["retrieval"], **(data.get("retrieval") or {})},
        "llm": {**defaults["llm"], **(data.get("llm") or {})},
    }


# Lazy-loaded index (shared across requests)
_index_state: Optional[Tuple[Any, List[dict], dict]] = None


def _get_index():
    global _index_state
    if _index_state is None:
        _index_state = build_index()
    return _index_state


def rag_query(
    question: str,
    top_k: int = None,
    max_tokens: int = None,
    config: Dict[str, Any] = None,
) -> Tuple[List[Tuple[int, str, float]], str]:
    cfg = config or _load_config()
    top_k = top_k or cfg.get("retrieval", {}).get("top_k", 5)
    llm_cfg = cfg.get("llm", {})
    max_tokens = max_tokens or llm_cfg.get("max_tokens", 512)
    temperature = llm_cfg.get("temperature", 0.2)
    model = llm_cfg.get("model", "llama-3.3-70b-versatile")

    bm25, corpus, id_to_passage = _get_index()
    passages = search(question, bm25, corpus, id_to_passage, k=top_k)
    context = format_passages_as_context(passages)
    prompt = build_rag_prompt(context=context, question=question)
    raw = generate(prompt, model=model, max_tokens=max_tokens, temperature=temperature)
    answer = parse_answer(raw)
    return passages, answer
