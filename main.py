from app.ingestion.loaders import load_pdf
from app.ingestion.splitter import split_documents
from app.ingestion.embedder import create_embedding_model

from app.vectorstore.store import create_vector_store

from app.retrieval.retriever import create_retriever

from app.generation.llm import create_llm
from app.generation.prompt import create_rag_prompt

from app.rag.chain import create_rag_chain


file_path = "data/raw/Emotion_Detection_Learning_Support_Engine_Internship_Report.pdf"


def main():

    # 1. Load documents
    documents = load_pdf(file_path)

    print(f"Loaded documents: {len(documents)}")

    # 2. Split documents
    chunks = split_documents(documents)

    print(f"Created chunks: {len(chunks)}")

    # 3. Create embedding model
    embedding_model = create_embedding_model()

    # 4. Create vector store
    vector_store = create_vector_store(
        chunks,
        embedding_model,
    )

    # 5. Create retriever
    retriever = create_retriever(vector_store)
    query = "How does the system detect emotions?"

    retrieved_docs = retriever.invoke(query)

    print("Number of retrieved documents:", len(retrieved_docs))

    print("\n==============================")
    print("RETRIEVED DOCUMENTS")
    print("==============================")

    for i, doc in enumerate(retrieved_docs):
        print(f"\n--- Retrieved Document {i + 1} ---")

        print("\nContent:")
        print(doc.page_content)

        print("\nMetadata:")
        print(doc.metadata)

    # 6. Create LLM
    llm = create_llm()

    # 7. Create prompt
    prompt = create_rag_prompt()

    # 8. Create RAG chain
    rag_chain = create_rag_chain(
        retriever,
        prompt,
        llm,
    )

    # 9. Ask question
    question = "How does the system detect emotions?"

    answer = rag_chain.invoke(question)

    print("\n ANSWER")
    print("==============================")

    print(answer)


if __name__ == "__main__":
    main()