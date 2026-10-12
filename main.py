from pathlib import Path
from app.ingestion.loaders import load_pdf
from app.ingestion.splitter import split_documents
from app.ingestion.embedder import create_embedding_model

from app.vectorstore.store import create_vector_store

from app.retrieval.retriever import create_retriever

from app.generation.llm import create_llm
from app.generation.prompt import create_rag_prompt

from app.rag.chain import create_rag_chain


data_directory = "data/raw"


def main():
    # 1. Load all PDFs
    documents = load_pdf(data_directory)

    print(f"Loaded pages: {len(documents)}")
    print(
        "PDF source files:",
        len({
            doc.metadata.get("source")
            for doc in documents
        }),
    )

    # 2. Split documents
    chunks = split_documents(documents)
    print(f"Created chunks: {len(chunks)}")

    # 3. Create embeddings
    embedding_model = create_embedding_model()

    # 4. Create vector store
    vector_store = create_vector_store(
        chunks,
        embedding_model,
    )

    # 5. Create retriever
    retriever = create_retriever(vector_store)

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

    # 9. Interactive question-answering loop
    print("\nResearch Insight Assistant is ready!")
    print("Ask questions about your PDFs.")
    print("Type 'exit' to stop.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            continue

        result = rag_chain.invoke(question)

        print("\nAssistant:")
        print(result["answer"])

        print("\nSources:")

        for i, doc in enumerate(
            result["documents"],
            start=1,
        ):
            source = Path(
                doc.metadata.get("source", "Unknown source")
            ).name

            page = doc.metadata.get(
                "page_label",
                doc.metadata.get("page", "Unknown"),
            )

            print(f"[Source {i}] {source} — Page {page}")

        print("\n" + "-" * 60)


if __name__ == "__main__":
    main()
