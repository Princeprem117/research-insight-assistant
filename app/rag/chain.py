
from pathlib import Path

from langchain_classic.chains import create_history_aware_retriever

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
            f"File Name: {source}\n"
            f"Page: {page}\n"
            f"Content:\n{doc.page_content}"
        )

    return "\n\n--\n\n".join(formatted)


def get_question_text(data):
    question = data.get("question")

    if isinstance(question, str):
        return question

    if isinstance(question, dict):
        nested_question = question.get("question")
        if isinstance(nested_question, str):
            return nested_question

    raise TypeError(
        f"Expected question to be a string, "
        f"but received {type(question).__name__}: {question!r}"
    )


def create_rag_chain(
    retriever: BaseRetriever,
    prompt: ChatPromptTemplate,
    llm: ChatOpenRouter,
    history_aware_prompt: ChatPromptTemplate,
):
    # 1. Build the query-rewriting chain.
    # We reproduce the standard history-aware behavior so
    # that the rewritten query can be printed for debugging.
    rewrite_chain = (
        history_aware_prompt
        | llm
        | StrOutputParser()
    )

    def retrieve_with_history(data):
        question = get_question_text(data)
        chat_history = data.get("chat_history", [])

        print("\n[DEBUG] Original question:", question)

        if chat_history:
            rewritten_question = rewrite_chain.invoke({
                "input": question,
                "chat_history": chat_history,
            })
        else:
            rewritten_question = question

        print("[DEBUG] Search query:", rewritten_question)

        documents = retriever.invoke(rewritten_question)

        print("[DEBUG] Retrieved documents:", len(documents))

        for i, doc in enumerate(documents, start=1):
            source = Path(
                doc.metadata.get("source", "Unknown source")
            ).name
            page = doc.metadata.get(
                "page_label",
                doc.metadata.get("page", "Unknown"),
            )

            print(
                f"[DEBUG] Source {i}: {source} | Page: {page}"
            )

        return documents

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
            documents=RunnableLambda(retrieve_with_history)
        )
        | RunnablePassthrough.assign(
            answer=answer_chain
        )
    )

    return rag_chain
