import re
import logging
from typing import Any, Dict, List, Sequence

from openai import OpenAI

from src.generation.llm import resolve_model

logger = logging.getLogger(__name__)

# Each evidence passage is trimmed before it goes into the judge prompt so a
# long-tail passage cannot push the request over the provider's size limit.
_MAX_CHARS_PER_PASSAGE = 1500


def score_groundedness(
    answer: str,
    evidence: Sequence[Dict[str, Any]],
    llm_client: OpenAI,
    model: str = None,
) -> float:
    """Score how well an answer is supported by the passages it was built from.

    The judge is shown the answer *and the full text of each evidence passage*,
    because a grader that only sees citation markers has no way to tell whether
    a claim is supported -- it can only guess from the shape of the answer.

    On LLM errors, returns 0.5 (neutral fallback) to avoid failing requests.

    Args:
        answer: The generated answer text.
        evidence: Passages the answer should be grounded in. Each item is a dict
            with a "text" key, optionally "marker" and "passage_id" (the shape
            produced by rag_query_full's citation list).
        llm_client: OpenAI client for calling the LLM.
        model: Optional model override; resolved from the environment if unset.

    Returns:
        float: Groundedness score between 0 and 1.
    """
    if not llm_client:
        logger.warning("LLM client is None, returning fallback groundedness score 0.5")
        return 0.5

    if not evidence:
        # Nothing to be grounded in, so the answer cannot be supported. Saying
        # so costs nothing and is more honest than asking the judge to guess.
        logger.warning("No evidence passages supplied; scoring answer as ungrounded")
        return 0.0

    prompt = _build_prompt(answer, evidence)

    try:
        response = llm_client.chat.completions.create(
            model=resolve_model(model),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.0,
        )
        return _parse_score_from_response(response.choices[0].message.content or "")
    except Exception as e:
        logger.error(f"Error computing groundedness score: {e}")
        return 0.5  # Fallback to neutral score on error


def _build_prompt(answer: str, evidence: Sequence[Dict[str, Any]]) -> str:
    blocks: List[str] = []
    for i, passage in enumerate(evidence, 1):
        marker = passage.get("marker") or f"[Passage {i}]"
        text = (passage.get("text") or "").strip()
        if len(text) > _MAX_CHARS_PER_PASSAGE:
            text = text[:_MAX_CHARS_PER_PASSAGE] + " ...[truncated]"
        pid = passage.get("passage_id")
        header = f"{marker} (id={pid})" if pid is not None else marker
        blocks.append(f"{header}\n{text}")
    passages_block = "\n\n".join(blocks)

    return f"""You are grading whether an answer is supported by its source passages.

Source passages:
{passages_block}

Answer to grade:
{answer}

Compare every factual claim in the answer against the passages above. Rate the
groundedness on a scale of 0.0 to 1.0 where:
- 1.0 = every claim is directly supported by the passages
- 0.7 = most claims are supported, minor unsupported details
- 0.5 = some claims are supported, some are not
- 0.3 = few claims are supported, most are not in the passages
- 0.0 = the answer contradicts the passages or is unsupported by them

Respond with ONLY the numerical score (e.g., 0.85). Do not include any other text."""


def _parse_score_from_response(response_text: str) -> float:
    """Extract a numerical score (0-1) from LLM response text.

    Tries multiple patterns:
    - Direct decimal: "0.85"
    - With label: "score: 0.75" or "Groundedness score: 0.80"
    - Integer: "1" or "0"

    Args:
        response_text: The LLM response text.

    Returns:
        float: Parsed score between 0 and 1, or 0.5 if parsing fails.
    """
    response_text = response_text.strip()

    # Try to find a decimal number
    decimal_match = re.search(r"0\.\d+", response_text)
    if decimal_match:
        try:
            score = float(decimal_match.group())
            # Clamp to [0, 1]
            return max(0.0, min(1.0, score))
        except ValueError:
            pass

    # Try to find any number and normalize
    number_match = re.search(r"\d+(?:\.\d+)?", response_text)
    if number_match:
        try:
            num = float(number_match.group())
            # If it's 0 or 1, treat as integer score
            if num in (0, 1):
                return float(num)
            # If it's between 0 and 1, use as-is
            if 0 <= num <= 1:
                return num
            # If it's > 1, assume it was 0-100 scale, normalize
            if num > 1:
                return min(1.0, num / 100.0)
        except ValueError:
            pass

    logger.warning(f"Could not parse groundedness score from: {response_text}")
    return 0.5
