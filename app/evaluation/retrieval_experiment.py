
from app.ingestion.embedder import create_embedding_model
from app.vectorstore.store import create_vector_store

def main():
    embedding_model = create_embedding_model()
    vector_store = create_vector_store(embedding_model)

    question = (
        "How does the Deep Agents implementation "
        "perform vector similarity search?"
    )

    configurations = {
        "Similarity": {
            "search_type": "similarity",
            "search_kwargs": {"k": 5},
        },
        "MMR": {
            "search_type": "mmr",
            "search_kwargs":{
                "k": 5,
                "fetch_k": 10,
                "lambda_mult": 0.5,
            },
        },
    }

    for name, config in configurations.items():
        retriever = vector_store.as_retriever(**config)
        documents = retriever.invoke(question)

        print(f"\n{'=' * 60}")
        print(f"{name} results")
        print(f"{'=' * 60}")

        for rank, doc in enumerate(documents, start=1):
            print(f"\nRank: {rank}")
            print(f"Source: {doc.metadata.get('source')}")
            print(f"Page: {doc.metadata.get('page_label', doc.metadata.get('page'))}")
            print(f"Content:\n{doc.page_content}")


if __name__ == "__main__":
    main()
