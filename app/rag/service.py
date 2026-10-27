
import hashlib
from pathlib import Path

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

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


class RAGService:
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.chat_histories = {}

        # Initialize shared RAG components.
        embeddings = create_embedding_model()
        self.vector_store = create_vector_store(embeddings)

        # Synchronize new, modified, and deleted PDFs.
        self.sync_index()

        retriever = create_retriever(self.vector_store)
        llm = create_llm()

        rag_chain = create_rag_chain(
            retriever,
            create_rag_prompt(),
            llm,
            history_aware_prompt=create_history_aware_prompt(),
        )

        self.chat = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def get_session_history(self, session_id: str):
        if session_id not in self.chat_histories:
            self.chat_histories[session_id] = (
                InMemoryChatMessageHistory()
            )
        return self.chat_histories[session_id]

    @staticmethod
    def file_hash(path: Path) -> str:
        hasher = hashlib.sha256()

        with path.open("rb") as file:
            for block in iter(
                lambda: file.read(1024 * 1024), b""
            ):
                hasher.update(block)

        return hasher.hexdigest()

    def sync_index(self):
        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"PDF directory not found: {self.data_dir}"
            )

        pdfs = sorted(
            p.resolve()
            for p in self.data_dir.glob("*.pdf")
            if p.is_file()
        )

        if not pdfs:
            raise ValueError(
                f"No PDFs found in {self.data_dir}"
            )

        indexed = get_indexed_file_hashes(self.vector_store)
        current = {str(p) for p in pdfs}

        # Remove chunks for deleted PDFs.
        for source in set(indexed) - current:
            ids = get_source_chunk_ids(
                self.vector_store, source
            )
            delete_chunks_by_ids(self.vector_store, ids)
            print(f"[INDEX] Deleted: {Path(source).name}")

        # Add new PDFs and update changed PDFs.
        for pdf in pdfs:
            source = str(pdf)
            digest = self.file_hash(pdf)

            if indexed.get(source) == digest:
                print(f"[INDEX] Unchanged: {pdf.name}")
                continue

            old_ids = get_source_chunk_ids(
                self.vector_store, source
            )

            documents = load_pdf_file(pdf)
            chunks = split_documents(documents)

            add_pdf_chunks(
                self.vector_store,
                chunks,
                source,
                digest,
            )

            # Delete old chunks after new chunks are added.
            delete_chunks_by_ids(
                self.vector_store, old_ids
            )

            action = (
                "Updated" if source in indexed else "Added"
            )
            print(
                f"[INDEX] {action}: {pdf.name} "
                f"({len(chunks)} chunks)"
            )

    def ask(self, question: str, session_id: str = "default"):
        return self.chat.invoke(
            {"question": question},
            config={
                "configurable": {
                    "session_id": session_id
                }
            },
        )
