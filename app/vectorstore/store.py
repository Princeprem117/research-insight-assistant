
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

COLLECTION_NAME = "research_insight"
PERSIST_DIRECTORY = "data/chroma"

def create_vector_store(
    embedding_model: HuggingFaceEmbeddings,
) -> Chroma:
    """Open the persistent Chroma collection without deleting it."""
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=PERSIST_DIRECTORY,
    )

    # One-time migration: old chunks may not have tracking metadata.
    existing = vector_store.get(include=["metadatas"])
    existing_ids = existing.get("ids", [])
    metadatas = existing.get("metadatas") or []

    if existing_ids and any(
        not metadata
        or not metadata.get("source_path")
        or not metadata.get("file_hash")
        for metadata in metadatas
    ):
        print(
            "[INDEX] Existing chunks lack tracking metadata. "
            "Rebuilding the collection once."
        )

        vector_store.delete_collection()

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embedding_model,
            persist_directory=PERSIST_DIRECTORY,
        )

    return vector_store


def get_indexed_file_hashes(
    vector_store: Chroma,
) -> dict[str, str]:
    """Return the known content hash for each indexed PDF."""
    result = vector_store.get(include=["metadatas"])

    indexed_files = {}

    for metadata in result.get("metadatas") or []:
        if not metadata:
            continue

        source_path = metadata.get("source_path")
        file_hash = metadata.get("file_hash")

        if source_path and file_hash:
            indexed_files[source_path] = file_hash

    return indexed_files


def get_source_chunk_ids(
    vector_store: Chroma,
    source_path: str,
) -> list[str]:
    """Find all chunk IDs belonging to a PDF."""
    result = vector_store.get(
        where={"source_path": source_path},
        include=["metadatas"],
    )

    return result.get("ids", [])


def add_pdf_chunks(
    vector_store: Chroma,
    chunks: list[Document],
    source_path: str,
    file_hash: str,
) -> None:
    """Add chunks with stable IDs and source-tracking metadata."""
    if not chunks:
        raise ValueError(
            f"No chunks generated for: {source_path}"
        )

    import hashlib

    source_id = hashlib.sha256(
        source_path.encode("utf-8")
    ).hexdigest()[:16]

    ids = []

    for index, chunk in enumerate(chunks):
        chunk.metadata["source_path"] = source_path
        chunk.metadata["file_hash"] = file_hash

        # Include the file hash so changed versions get new IDs.
        ids.append(
            f"{source_id}-{file_hash[:24]}-{index}"
        )

    vector_store.add_documents(
        documents=chunks,
        ids=ids,
    )


def delete_chunks_by_ids(
    vector_store: Chroma,
    chunk_ids: list[str],
) -> None:
    """Delete specific chunks from Chroma."""
    if chunk_ids:
        vector_store.delete(ids=chunk_ids)
