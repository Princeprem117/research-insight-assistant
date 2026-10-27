
from pathlib import Path

from app.rag.service import RAGService


def main():
    # Initialize the shared RAG service once.
    rag_service = RAGService()

    print(
        "\nResearch Insight Assistant ready! "
        "Type 'exit' to stop."
    )

    while True:
        question = input("\nYOU: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("\nGoodbye!")
            break

        if not question:
            continue

        result = rag_service.ask(
            question=question,
            session_id="terminal-session",
        )

        print("\nASSISTANT:\n", result["answer"])
        print("\nSources:")
        print("-" * 30)

        for i, doc in enumerate(
            result.get("documents", []), start=1
        ):
            source = Path(
                doc.metadata.get(
                    "source",
                    doc.metadata.get("source_path", "Unknown"),
                )
            ).name

            page = doc.metadata.get(
                "page_label",
                doc.metadata.get("page", "?"),
            )

            print(f"[Source {i}] {source} — Page {page}")


if __name__ == "__main__":
    main()
