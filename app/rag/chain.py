from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
)
from langchain_openrouter import ChatOpenRouter


def create_rag_chain(
    retriever: BaseRetriever,
    prompt: ChatPromptTemplate,
    llm: ChatOpenRouter,
):
    answer_chain = (
        {
            "context": lambda x: x["documents"],
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain = (
        RunnablePassthrough.assign(
            documents= lambda x: retriever.invoke(x["question"]),
        )
        | RunnableParallel(
            answer=answer_chain,
            documents=lambda x: x["documents"],
        )
    )

    return rag_chain
