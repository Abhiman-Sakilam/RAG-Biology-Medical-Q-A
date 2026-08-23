import re
import logging
from typing import List, Tuple, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)


def score_groundedness(
    answer: str,
    citations: List[Tuple[str, int]],
    llm_client: OpenAI,
    model: str = None,
) -> float:
    """Score the groundedness of an answer based on its citations.

    Asks an LLM to evaluate whether the answer is properly grounded in the cited passages.
    Returns a score between 0 and 1 where higher means more grounded.

    On LLM errors, returns 0.5 (neutral fallback) to avoid failing requests.

    Args:
        answer: The generated answer text.
        citations: List of (marker, passage_id) tuples for cited passages.
        llm_client: OpenAI client for calling the LLM.
        model: Optional model override; uses default from env if not specified.

    Returns:
        float: Groundedness score between 0 and 1.
    """
    if not llm_client:
        logger.warning("LLM client is None, returning fallback groundedness score 0.5")
        return 0.5

    # Build the groundedness evaluation prompt
    citations_info = "\n".join(
        [f"- {marker} (Passage ID: {pid})" for marker, pid in citations]
    )

    prompt = f"""Evaluate the groundedness of the following answer based on the citations provided.

Answer:
{answer}

Citations in the answer:
{citations_info if citations_info else "No citations found"}

Rate the groundedness on a scale of 0.0 to 1.0 where:
- 1.0 = All claims are fully supported by citations and passages
- 0.7 = Most claims are grounded, minor gaps exist
- 0.5 = Some claims are supported, some are not
- 0.3 = Few claims are supported, many are unsupported
- 0.0 = No claims are grounded in passages or no citations provided

Respond with ONLY the numerical score (e.g., 0.85). Do not include any other text."""

    try:
        response = llm_client.chat.completions.create(
            model=model or "gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.0,
        )
        response_text = response.choices[0].message.content or ""

        # Extract the numerical score from the response
        score = _parse_score_from_response(response_text)
        return score

    except Exception as e:
        logger.error(f"Error computing groundedness score: {e}")
        return 0.5  # Fallback to neutral score on error


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
