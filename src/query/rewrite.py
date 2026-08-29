import logging
from typing import Tuple

from src.generation.llm import resolve_model

logger = logging.getLogger(__name__)


def rewrite_query(question: str, llm_client, model: str = None) -> Tuple[str, str]:
    """Generate a hypothetical passage and keyword expansions for the question using HyDE.

    Args:
        question: The user's question to rewrite.
        llm_client: An LLM client with chat_completions_create method.
        model: Chat model id. Defaults to the configured provider model
            (see src.generation.llm.resolve_model).

    Returns:
        Tuple of (hypothesis_passage, keyword_expansions).
        On error, returns (question, "").
    """
    system_prompt = (
        "You are a helpful assistant that generates hypothetical passages and keyword expansions. "
        "For the given question, generate a hypothetical passage that would answer it, "
        "and provide 2-3 synonym/expansion keywords separated by commas."
    )

    user_prompt = (
        f"Question: {question}\n\n"
        "Provide your response in this format:\n"
        "Hypothesis: [a hypothetical passage that answers this question]\n"
        "Expansions: [2-3 keyword synonyms or related terms, comma-separated]"
    )

    try:
        response = llm_client.chat.completions.create(
            model=resolve_model(model),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=256,
            temperature=0.3,
        )

        content = response.choices[0].message.content
        if not content:
            logger.warning("Empty response from LLM for query rewriting")
            return question, ""

        # Parse hypothesis and expansions from the response
        hypothesis = ""
        expansions = ""

        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("Hypothesis:"):
                hypothesis = line.replace("Hypothesis:", "").strip()
            elif line.startswith("Expansions:"):
                expansions = line.replace("Expansions:", "").strip()

        # Fallback if parsing didn't find anything
        if not hypothesis:
            hypothesis = question
        if not expansions:
            expansions = ""

        return hypothesis, expansions

    except Exception as e:
        logger.warning("Query rewriting failed (%s); using original question", e)
        return question, ""
