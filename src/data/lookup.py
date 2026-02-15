from typing import Dict, List, Any


def build_lookup(corpus: List[Dict[str, Any]]) -> Dict[int, str]:
    return {item["id"]: item["passage"] for item in corpus}
