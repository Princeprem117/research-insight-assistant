from langchain_huggingface import HuggingFaceEmbeddings


def create_embedding_model():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    return embeddings