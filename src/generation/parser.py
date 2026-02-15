import re


def parse_answer(raw: str, max_length: int = 2000) -> str:
    if not raw:
        return ""
    text = raw.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) > max_length:
        text = text[:max_length] + "..."
    return text
