import re
import logging
from typing import Tuple, List, Set

logger = logging.getLogger(__name__)


def parse_answer(raw: str, max_length: int = 2000) -> str:
    if not raw:
        return ""
    text = raw.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) > max_length:
        text = text[:max_length] + "..."
    return text


def extract_and_validate_citations(
    answer: str, retrieved_passage_ids: Set[int]
) -> Tuple[str, List[Tuple[str, int]]]:
    """Extract and validate citations from an answer.

    Finds [Passage N] or [P N] markers in the answer text.
    Validates that cited passage IDs are in the retrieved set.
    Strips invalid citations and returns cleaned answer + valid citations.

    Args:
        answer: The raw answer text containing potential citations.
        retrieved_passage_ids: Set of passage IDs that were actually retrieved.

    Returns:
        Tuple of (cleaned_answer, citations_list) where:
        - cleaned_answer: Answer with invalid citations removed
        - citations_list: List of (marker, passage_id) tuples for valid citations
    """
    # Regex patterns: [Passage N] or [P N] where N is one or more digits
    pattern = r"\[(?:Passage|P)\s+(\d+)\]"

    citations = []
    invalid_markers = set()
    cleaned_answer = answer

    # Find all citation markers
    for match in re.finditer(pattern, answer):
        marker = match.group(0)  # Full marker like "[Passage 1]" or "[P 2]"
        passage_id = int(match.group(1))  # Extract the ID

        if passage_id in retrieved_passage_ids:
            citations.append((marker, passage_id))
        else:
            # Mark invalid citations for removal
            invalid_markers.add(marker)
            logger.warning(f"Invalid citation: {marker} (passage ID {passage_id} not in retrieved set)")

    # Remove invalid citations from the answer
    for marker in invalid_markers:
        cleaned_answer = cleaned_answer.replace(marker, "")

    return cleaned_answer, citations
