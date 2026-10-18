
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder


def create_rag_prompt():
    prompt = ChatPromptTemplate.from_messages([
        (
            '''
            "system",
            """
    You are a research assistant.

    Answer the user's question using only the
    provided research context.

    The context contains source labels, PDF filenames,
    page numbers, and document excerpts.

    Rules:
    1. Use only evidence found in the provided context.
    2. Cite factual claims using labels such as [Source 1].
    3. Use the labels exactly as provided.
    4. Never invent citations, filenames, or page numbers.
    5. If the context does not contain enough evidence,
    clearly say that you could not find the answer
    in the provided research documents.
    6. Distinguish facts from reasonable inferences.

    Research context:


    {context}
    """
    '''
                "system",
            """
Given the conversation history and the latest user
question, rewrite the question into a standalone
question that can be understood without the history.

Rules:
- Resolve references such as "it", "that", and "the
  second pattern" using the conversation history.
- Preserve the user's original intent.
- Do not answer the question.
- Do not add facts that are not present in the question
  or conversation.
- If the question is already standalone, return it
  unchanged.
- Output only the rewritten question.
"""
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
    ])

    return prompt