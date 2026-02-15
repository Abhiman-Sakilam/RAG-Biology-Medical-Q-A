SYSTEM_TEMPLATE = """You are a helpful assistant that answers questions based only on the provided context passages (biology/medical domain). If the context does not contain enough information, say so briefly. Keep answers concise and factual."""

USER_TEMPLATE = """Use the following passages to answer the question.

Context:
{context}

Question: {question}

Answer:"""


def build_rag_prompt(
    context: str,
    question: str,
    include_system: bool = True,
) -> str:
    user_msg = USER_TEMPLATE.format(context=context, question=question)
    if include_system:
        return f"{SYSTEM_TEMPLATE.strip()}\n\n{user_msg}"
    return user_msg
