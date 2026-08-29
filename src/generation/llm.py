import os

from openai import OpenAI

from src.config.env import load_env

load_env()

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"


def resolve_model(model: str = None) -> str:
    """Resolve the chat model id to use.

    Every LLM call in the pipeline (generation, query rewriting, groundedness
    scoring) goes through here so no caller has to hardcode a model id that
    may not exist on the configured provider.
    """
    return (
        model
        or os.getenv("GROQ_MODEL")
        or os.getenv("OPENAI_MODEL")
        or DEFAULT_MODEL
    )


def _get_client() -> OpenAI:
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    api_key = groq_key or openai_key or os.getenv("LLM_API_KEY")

    if not api_key:
        raise ValueError(
            "Set GROQ_API_KEY (or OPENAI_API_KEY) in setup/.env (see setup/.env.example)"
        )

    if groq_key:
        base_url = GROQ_BASE_URL
    elif os.getenv("OPENAI_BASE_URL"):
        base_url = os.getenv("OPENAI_BASE_URL")
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
    model = resolve_model(model)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""
