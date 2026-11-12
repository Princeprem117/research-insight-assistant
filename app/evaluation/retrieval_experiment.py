
import csv
from pathlib import Path

from app.ingestion.embedder import create_embedding_model
from app.vectorstore.store import create_vector_store


OUTPUT_FILE = Path("evaluation_results/reformulated_retrieval.csv")


REFORMULATED_QUERIES = {
    "Where does retrieval take place in a RAG pipeline?":
        "RAG pipeline retrieval step: where are relevant document chunks retrieved from storage?",

    "What is agentic RAG?":
        "Definition, architecture, and key characteristics of Agentic Retrieval-Augmented Generation",

    "How does the Deep Agents implementation perform vector similarity search?":
        "Deep Agents VectorStore embeddings similarity search implementation",

    "How do embeddings support retrieval in LangChain?":
        "How LangChain embedding models convert document chunks and queries into vectors for semantic similarity search",

    "What RAG approaches are discussed in the agentic RAG survey?":
        "RAG approaches, architectures, and retrieval strategies discussed in the Agentic RAG survey",
}


def get_relevance():
    while True:
        answer = input("\nIs this chunk relevant? (y/n): ").strip().lower()

        if answer in ("y", "n"):
            return "yes" if answer == "y" else "no"

        print("Please enter only y or n.")


def main():
    embedding_model = create_embedding_model()
    vector_store = create_vector_store(embedding_model)

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 10,
            "lambda_mult": 0.5,
        },
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "question",
        "search_type",
        "k",
        "rank",
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

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for question_number, (question, query) in enumerate(
            REFORMULATED_QUERIES.items(), start=1
        ):
            documents = retriever.invoke(query)

            print("\n" + "=" * 70)
            print(f"QUESTION {question_number}/5")
            print(f"Original question: {question}")
            print(f"Reformulated query: {query}")
            print("=" * 70)

            for rank, doc in enumerate(documents, start=1):
                source = Path(
                    doc.metadata.get("source", "Unknown")
                ).name

                page = doc.metadata.get(
                    "page_label",
                    doc.metadata.get("page", "Unknown"),
                )

                print(f"\n--- Retrieved Chunk {rank}/3 ---")
                print(f"Source: {source}")
                print(f"Page: {page}")
                print(f"Content:\n{doc.page_content}")

                relevance = get_relevance()

                writer.writerow({
                    "question": question,
                    "search_type": "mmr",
                    "k": 3,
                    "rank": rank,
                    "source": source,
                    "page": page,
                    "content": doc.page_content,
                    "relevant": relevance,
                })

                csvfile.flush()

        print("\nEvaluation completed!")
        print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
