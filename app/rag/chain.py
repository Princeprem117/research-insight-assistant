from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableParallel , RunnablePassthrough
from langchain_openrouter import ChatOpenRouter


def create_rag_chain(
    retriever: BaseRetriever,
    prompt: ChatPromptTemplate,
    llm: ChatOpenRouter,
):

    answer_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain = RunnableParallel(
        answer = answer_chain,
        documents = retriever,
    )

    return rag_chain