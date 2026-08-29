from typing import List, Tuple


def format_passages_as_context(
    passages: List[Tuple[int, str, float]],
    max_chars: int = 6000,
    include_scores: bool = False,
) -> str:
    parts = []
    total = 0
    for i, (pid, text, score) in enumerate(passages, 1):
        line = f"[Passage {i}] (id={pid})" if include_scores else f"[Passage {i}]"
        if include_scores:
            line += f" (score={score:.2f})"
        line += f"\n{text.strip()}\n"
        if total + len(line) > max_chars:
            remaining = max_chars - total - 20
            if remaining > 0:
                parts.append(line[:remaining] + "\n...[truncated]\n")
            # When the budget is already spent there is nothing left to add;
            # appending the untrimmed line here would blow past max_chars.
            break
        parts.append(line)
        total += len(line)
    return "\n".join(parts)
