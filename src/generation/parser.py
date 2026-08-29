import re
import logging
from typing import List, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

# [Passage N] or [P N]. N is the passage's 1-based position in the prompt
# context, not its corpus id -- see extract_and_validate_citations.
_CITATION_PATTERN = re.compile(r"\[(?:Passage|P)\s+(\d+)\]")


def parse_answer(raw: str, max_length: int = 2000) -> str:
    if not raw:
        return ""
    text = raw.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) > max_length:
        text = text[:max_length] + "..."
    return text


def extract_and_validate_citations(
    answer: str, ordered_passage_ids: Sequence[Union[int, str]]
) -> Tuple[str, List[Tuple[str, Union[int, str]]]]:
    """Extract citation markers from an answer and resolve them to passage ids.

    `format_passages_as_context` labels the context by 1-based position
    ([Passage 1] .. [Passage N]), so the number the model writes is an ordinal
    into the retrieved list -- never a corpus id. A marker is valid when that
    ordinal addresses a passage that was actually retrieved, and it resolves to
    that passage's real id, which may be an int or a "parent::chunk" string.

    Args:
        answer: The raw answer text containing potential citations.
        ordered_passage_ids: Ids of the retrieved passages, in the same order
            they were rendered into the prompt context.

    Returns:
        Tuple of (cleaned_answer, citations_list) where:
        - cleaned_answer: Answer with out-of-range citations removed
        - citations_list: (marker, passage_id) tuples in order of appearance,
          duplicates preserved
    """
    citations: List[Tuple[str, Union[int, str]]] = []
    invalid_markers = set()

    for match in _CITATION_PATTERN.finditer(answer):
        marker = match.group(0)
        ordinal = int(match.group(1))
        if 1 <= ordinal <= len(ordered_passage_ids):
            citations.append((marker, ordered_passage_ids[ordinal - 1]))
        else:
            invalid_markers.add(marker)
            logger.warning(
                "Invalid citation: %s (only %d passages were retrieved)",
                marker,
                len(ordered_passage_ids),
            )

    cleaned_answer = answer
    for marker in invalid_markers:
        cleaned_answer = cleaned_answer.replace(marker, "")
    if invalid_markers:
        cleaned_answer = _tidy_whitespace(cleaned_answer)

    return cleaned_answer, citations


def _tidy_whitespace(text: str) -> str:
    """Repair the spacing left behind when a marker is spliced out mid-sentence."""
    text = re.sub(r" +([.,;:!?])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()
