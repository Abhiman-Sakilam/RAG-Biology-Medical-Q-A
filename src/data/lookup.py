from typing import Dict, List, Any, Union


def build_lookup(corpus: List[Dict[str, Any]]) -> Dict[Union[int, str], str]:
    """Build lookup from chunk/passage id to text.

    Args:
        corpus: List of dicts with at least id and passage/text keys.
                Handles both chunked (id="parent::idx") and non-chunked ids.

    Returns:
        Mapping from id to passage text.
    """
    return {item["id"]: item.get("passage", item.get("text", "")) for item in corpus}
