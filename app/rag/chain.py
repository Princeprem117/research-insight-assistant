
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_openrouter import ChatOpenRouter


def format_docs(documents):
    formatted = []

    for i, doc in enumerate(documents, start=1):
        source = Path(
            doc.metadata.get("source", "Unknown source")
        ).name

        page = doc.metadata.get(
            "page_label",
            doc.metadata.get("page", "Unknown"),
        )

        formatted.append(
            f"[Source {i}]\n"
            f"File: {source}\n"
            f"Page: {page}\n"
            f"Content:\n{doc.page_content}"
        )

    return "\n\n".join(formatted)


def create_rag_chain(
    retriever: BaseRetriever,
    prompt: ChatPromptTemplate,
    llm: ChatOpenRouter,
):

    answer_chain = (
        {
            "context": RunnableLambda(
                lambda x: format_docs(x["documents"])
            ),
            "question": RunnableLambda(get_question_text),
            "chat_history": RunnableLambda(
                lambda x: x["chat_history"]
            ),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain = (
        RunnablePassthrough.assign(
            documents=RunnableLambda(
                lambda x: retriever.invoke(get_question_text(x))
            )
        )
        | RunnablePassthrough.assign(
            answer=answer_chain
        )
    )

    return rag_chain

def get_question_text(data):
    question = data.get("question")

    # Normal case: question is already a string.
    if isinstance(question, str):
        return question

    # Handle an accidentally nested question dictionary.
    if isinstance(question, dict):
        nested_question = question.get("question")
        if isinstance(nested_question, str):
            return nested_question

    raise TypeError(
        f"Expected question to be a string, "
        f"but received {type(question).__name__}: {question!r}"
    )