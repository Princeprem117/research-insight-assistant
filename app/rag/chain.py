from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_openrouter import ChatOpenRouter


def create_rag_chain(
    retriever: BaseRetriever,
    prompt: ChatPromptTemplate,
    llm: ChatOpenRouter,
):

    chain = (
        {
            "context": retriever,
            "question": lambda x: x,
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain