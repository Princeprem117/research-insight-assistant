from langchain_core.retrievers import BaseRetriever
from langchain_chroma import Chroma

def create_retriever(vector_store: Chroma) -> BaseRetriever:
    retriever = vector_store.as_retriever(
        search_type = "mmr", 
        search_kwargs = {"k": 3,
                         "fetch_k": 10,
                         },
    )

    return retriever