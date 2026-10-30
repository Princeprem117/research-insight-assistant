
import hashlib
import json
from pathlib import Path

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
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
from app.rag.conversation_store import ConversationStore


class RAGService:
    def __init__(self, data_dir="data/raw"):
        self.data_dir = Path(data_dir)
        self.conversation_store = ConversationStore()
        self.chat_histories = {}

        embeddings = create_embedding_model()
        self.vector_store = create_vector_store(embeddings)
        self.sync_index()

        rag_chain = create_rag_chain(
            create_retriever(self.vector_store),
            create_rag_prompt(),
            create_llm(),
            history_aware_prompt=create_history_aware_prompt(),
        )

        self.chat = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def get_session_history(self, session_id):
        if session_id not in self.chat_histories:
            history = InMemoryChatMessageHistory()

            # Restore prior messages from SQLite.
            for row in self.conversation_store.get_messages(session_id):
                if row["role"] == "user":
                    history.add_message(
                        HumanMessage(content=row["content"])
                    )
                elif row["role"] == "assistant":
                    history.add_message(
                        AIMessage(content=row["content"])
                    )

            self.chat_histories[session_id] = history

        return self.chat_histories[session_id]

    @staticmethod
    def file_hash(path):
        hasher = hashlib.sha256()
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
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
            raise ValueError(f"No PDFs found in {self.data_dir}")

        indexed = get_indexed_file_hashes(self.vector_store)
        current = {str(p) for p in pdfs}

        for source in set(indexed) - current:
            delete_chunks_by_ids(
                self.vector_store,
                get_source_chunk_ids(self.vector_store, source),
            )
            print(f"[INDEX] Deleted: {Path(source).name}")

        for pdf in pdfs:
            source = str(pdf)
            digest = self.file_hash(pdf)

            if indexed.get(source) == digest:
                print(f"[INDEX] Unchanged: {pdf.name}")
                continue

            old_ids = get_source_chunk_ids(self.vector_store, source)
            documents = load_pdf_file(pdf)
            chunks = split_documents(documents)

            add_pdf_chunks(
                self.vector_store, chunks, source, digest
            )
            delete_chunks_by_ids(self.vector_store, old_ids)

            action = "Updated" if source in indexed else "Added"
            print(f"[INDEX] {action}: {pdf.name} ({len(chunks)} chunks)")

    def create_conversation(self, conversation_id):
        self.conversation_store.create_conversation(conversation_id)

    def list_conversations(self):
        return self.conversation_store.list_conversations()

    def get_conversation_messages(self, conversation_id):
        return self.conversation_store.get_messages(conversation_id)

    def ask(self, question, session_id="default"):
        if not self.conversation_store.get_conversation(session_id):
            self.conversation_store.create_conversation(session_id)

        history = self.get_session_history(session_id)
        is_first_question = len(history.messages) == 0

        result = self.chat.invoke(
            {"question": question},
            config={"configurable": {"session_id": session_id}},
        )

        # Save only the new turn; history already contains it in memory.
        sources = []
        seen = set()

        for doc in result.get("documents", []):
            file_name = Path(
                doc.metadata.get(
                    "source",
                    doc.metadata.get("source_path", "Unknown"),
                )
            ).name
            page = str(
                doc.metadata.get(
                    "page_label",
                    doc.metadata.get("page", "Unknown"),
                )
            )
            if (file_name, page) not in seen:
                seen.add((file_name, page))
                sources.append({"file": file_name, "page": page})

        self.conversation_store.add_message(
            session_id, "user", question
        )
        self.conversation_store.add_message(
            session_id, "assistant", result["answer"], sources
        )

        if is_first_question:
            title = question.strip().replace("\n", " ")[:40]
            if len(question.strip()) > 40:
                title += "..."
            self.conversation_store.update_title(session_id, title)

        return result
