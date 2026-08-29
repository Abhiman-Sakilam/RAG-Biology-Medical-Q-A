SYSTEM_TEMPLATE = """You are a helpful assistant that answers questions based only on the provided context passages (biology/medical domain). If the context does not contain enough information, say so briefly. Keep answers concise and factual."""

CITATION_INSTRUCTION = """Support every factual claim with a citation to the passage it came from, using the bracketed label exactly as it appears in the context (for example [Passage 2]). Cite only passages shown in the context, and do not invent labels for passages that are not there."""

USER_TEMPLATE = """Use the following passages to answer the question.

Context:
{context}

Question: {question}

Answer:"""


def build_rag_prompt(
    context: str,
    question: str,
    include_system: bool = True,
    require_citations: bool = False,
) -> str:
    """Assemble the RAG prompt.

    When `require_citations` is set (driven by `[citation] enforce` in
    config.toml) the model is explicitly asked to emit [Passage N] markers;
    without it the model is never told citations are wanted, so
    extract_and_validate_citations would find nothing to extract.
    """
    parts = []
    if include_system:
        system = SYSTEM_TEMPLATE.strip()
        if require_citations:
            system = f"{system}\n\n{CITATION_INSTRUCTION.strip()}"
        parts.append(system)
    elif require_citations:
        parts.append(CITATION_INSTRUCTION.strip())
    parts.append(USER_TEMPLATE.format(context=context, question=question))
    return "\n\n".join(parts)
