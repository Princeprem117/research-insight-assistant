import csv
import hashlib
from pathlib import Path

from app.ingestion.loaders import load_pdf
from app.ingestion.splitter import split_documents
from app.ingestion.embedder import create_embedding_model
from app.vectorstore.store import create_vector_store

from app.evaluation.test_questions import TEST_QUESTIONS


OUTPUT_DIR = Path("evaluation_results")
OUTPUT_FILE = OUTPUT_DIR / "retrieval_comparison.csv"


def chunk_id(document):
    content = document.page_content
    return hashlib.sha1(
        content.encode("utf-8")
    ).hexdigest()[:12]


def evaluate_retrieval():
    # Build the index using your existing pipeline.
    documents = load_pdf("data/raw")
    chunks = split_documents(documents)

    embedding_model = create_embedding_model()
    vector_store = create_vector_store(
        chunks,
        embedding_model,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for question in TEST_QUESTIONS:
        for search_type in ("similarity", "mmr"):
            for k in (3, 5):

                if search_type == "similarity":
                    retrieved = vector_store.similarity_search(
                        question,
                        k=k,
                    )
                else:
                    retrieved = (
                        vector_store.max_marginal_relevance_search(
                            question,
                            k=k,
                            fetch_k=max(20, k * 4),
                        )
                    )

                for rank, document in enumerate(
                    retrieved, start=1
                ):
                    rows.append({
                        "question": question,
                        "search_type": search_type,
                        "k": k,
                        "rank": rank,
                        "chunk_id": chunk_id(document),
                        "source": Path(
                            document.metadata.get(
                                "source", "Unknown"
                            )
                        ).name,
                        "page": document.metadata.get(
                            "page_label",
                            document.metadata.get("page", ""),
                        ),
                        "content": document.page_content,
                        "relevant": "",
                    })

    fieldnames = [
        "question",
        "search_type",
        "k",
        "rank",
        "chunk_id",
        "source",
        "page",
        "content",
        "relevant",
    ]

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks indexed: {len(chunks)}")
    print(f"Retrieved results exported: {len(rows)}")
    print(f"Evaluation file: {OUTPUT_FILE}")


if __name__ == "__main__":
    evaluate_retrieval()