import hashlib
from pathlib import Path

from app.ingestion.loaders import load_pdf_file
from app.ingestion.splitter import split_documents
from app.ingestion.embedder import create_embedding_model
from app.vectorstore.store import (
    create_vector_store,
    get_indexed_file_hashes,
    get_source_chunk_ids,
    add_pdf_chunks,
    delete_chunks_by_ids,
)
from app.retrieval.retriever import create_retriever
from app.generation.llm import create_llm
from app.generation.prompt import (
    create_rag_prompt,
    create_history_aware_prompt,
)
from app.rag.chain import create_rag_chain

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


DATA_DIR = Path("data/raw")
chat_store = {}


def get_session_history(session_id):
    if session_id not in chat_store:
        chat_store[session_id] = InMemoryChatMessageHistory()
    return chat_store[session_id]


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sync_index(vector_store):
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Directory not found: {DATA_DIR}")

    pdfs = sorted(p.resolve() for p in DATA_DIR.glob("*.pdf"))
    if not pdfs:
        raise ValueError(f"No PDFs found in {DATA_DIR}")

    indexed = get_indexed_file_hashes(vector_store)
    current = {str(p) for p in pdfs}

    # Remove PDFs that no longer exist.
    for source in set(indexed) - current:
        delete_chunks_by_ids(
            vector_store,
            get_source_chunk_ids(vector_store, source),
        )
        print(f"[INDEX] Deleted: {Path(source).name}")

    # Add new PDFs and update changed PDFs.
    for pdf in pdfs:
        source = str(pdf)
        digest = file_hash(pdf)

        if indexed.get(source) == digest:
            print(f"[INDEX] Unchanged: {pdf.name}")
            continue

        old_ids = get_source_chunk_ids(vector_store, source)

        documents = load_pdf_file(pdf)
        chunks = split_documents(documents)

        add_pdf_chunks(vector_store, chunks, source, digest)
        delete_chunks_by_ids(vector_store, old_ids)

        action = "Updated" if source in indexed else "Added"
        print(f"[INDEX] {action}: {pdf.name} ({len(chunks)} chunks)")


def main():
    embeddings = create_embedding_model()
    vector_store = create_vector_store(embeddings)

    sync_index(vector_store)

    rag_chain = create_rag_chain(
        create_retriever(vector_store),
        create_rag_prompt(),
        create_llm(),
        history_aware_prompt=create_history_aware_prompt(),
    )

    chat = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    print("\nResearch Insight Assistant ready! Type 'exit' to stop.")

    while True:
        question = input("\nYou: ").strip()

        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        result = chat.invoke(
            {"question": question},
            config={"configurable": {"session_id": "research-session"}},
        )

        print("\nAssistant:\n", result["answer"])
        print("\nSources:")

        for i, doc in enumerate(result["documents"], 1):
            source = Path(doc.metadata.get("source", "Unknown")).name
            page = doc.metadata.get("page_label", doc.metadata.get("page", "?"))
            print(f"[Source {i}] {source} — Page {page}")


if __name__ == "__main__":
    main()
