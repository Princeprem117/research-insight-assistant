from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


def create_vector_store(
        chunks: list[Document], 
        embedding_model: HuggingFaceEmbeddings,
        ) :
            collection_name = "research_insight"

            # connect to existing collection
            vector_store = Chroma(
                collection_name=collection_name,
                embedding_function = embedding_model,
                persist_directory = "data/chroma",
            )

            #remove the old collection fro the current chunks
            vector_store.delete_collection()

            #create new collection for current chunks
            vector_store = Chroma.from_documents(
                documents = chunks,
                embedding = embedding_model,
                collection_name = collection_name,
                persist_directory = "data/chroma",
            )
            
            return vector_store