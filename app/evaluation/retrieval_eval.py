
import csv
import hashlib
from pathlib import Path

from app.ingestion.embedder import create_embedding_model
from app.vectorstore.store import create_vector_store
from app.evaluation.test_questions import TEST_QUESTIONS


OUTPUT_DIR = Path("evaluation_results")
OUTPUT_FILE = OUTPUT_DIR / "retrieval_comparison.csv"


def chunk_id(document):
    return hashlib.sha1(
        document.page_content.encode("utf-8")
    ).hexdigest()[:12]


def evaluate_retrieval():
    # Connect to the existing persistent Chroma collection.
    embedding_model = create_embedding_model()
    vector_store = create_vector_store(embedding_model)

    # Check that the index contains documents.
    collection = vector_store.get(include=["metadatas"])

    if not collection.get("ids"):
        raise ValueError(
            "Chroma index is empty. Run main.py first to index your PDFs."
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

                for rank, document in enumerate(retrieved, start=1):
                    rows.append({
                        "question": question,
                        "search_type": search_type,
                        "k": k,
                        "rank": rank,
                        "chunk_id": chunk_id(document),
                        "source": Path(
                            document.metadata.get(
                                "source",
                                document.metadata.get(
                                    "source_path", "Unknown"
                                ),
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

    print(f"Test questions: {len(TEST_QUESTIONS)}")
    print(f"Configurations: 4 per question")
    print(f"Retrieved chunks: {len(rows)}")
    print(f"Evaluation file: {OUTPUT_FILE}")


if __name__ == "__main__":
    evaluate_retrieval()
