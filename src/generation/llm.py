import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent.parent.parent / "setup" / ".env")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def _get_client() -> OpenAI:
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        raise ValueError(
            "Set GROQ_API_KEY (or OPENAI_API_KEY) in setup/.env (see setup/.env.example)"
        )
    if os.getenv("OPENAI_BASE_URL"):
        base_url = os.getenv("OPENAI_BASE_URL")
    elif os.getenv("GROQ_API_KEY") or (api_key and api_key.strip().startswith("gsk_")):
        base_url = GROQ_BASE_URL
    else:
        base_url = None
    return OpenAI(api_key=api_key, base_url=base_url)


def generate(
    prompt: str,
    model: str = None,
    max_tokens: int = 512,
    temperature: float = 0.2,
) -> str:
    client = _get_client()
    model = model or os.getenv("GROQ_MODEL") or os.getenv("OPENAI_MODEL", "llama-3.3-70b-versatile")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""
