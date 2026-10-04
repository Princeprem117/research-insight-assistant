from app.ingestion.loaders import load_pdf
from app.ingestion.splitter import split_documents
from app.ingestion.embedder import create_embedding_model
from app.vectorstore.store import create_vector_store
from app.retrieval.retriever import create_retriever


file_path = "data/raw/Emotion_Detection_Learning_Support_Engine_Internship_Report.pdf"


def main():

    # 1. Load
    documents = load_pdf(file_path)

    # 2. Split
    chunks = split_documents(documents)

    # 3. Embeddings
    embedding_model = create_embedding_model()

    # 4. Vector store
    vector_store = create_vector_store(
        chunks,
        embedding_model
    )
    # 5. Retriever
    retriever = create_retriever(vector_store)

    # 6. Query
    query = "How does the system detect emotions?"

    results = retriever.invoke(query)

    for i, result in enumerate(results):

        print(f"\n--- Result {i + 1} ---")
        print(result.page_content)
        print(result.metadata)


if __name__ == "__main__":
    main()