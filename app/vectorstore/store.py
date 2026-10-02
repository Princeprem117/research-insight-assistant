from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


def create_vector_store(
        chunks: list[Document], 
        embedding_model: HuggingFaceEmbeddings,
        ) :
            vector_store = Chroma.from_documents(
                documents = chunks,
                embedding = embedding_model,
                persist_directory = "data/chroma",
                )

            return vector_store