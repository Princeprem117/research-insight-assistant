from app.ingestion.loaders import load_pdf
from app.ingestion.splitter import split_documents
from app.ingestion.embedder import create_embedding_model

file_path = "data/raw/Emotion_Detection_Learning_Support_Engine_Internship_Report.pdf"

def main():

    # --------------------------------
    # 1. Load the PDF
    # --------------------------------

    documents = load_pdf(file_path)

    print("\n==============================")
    print("LOADED DOCUMENTS")
    print("==============================")

    print(f"Number of documents: {len(documents)}")

    for i, document in enumerate(documents[:2]):

        print(f"\n--- Document {i} ---")

        print("Type:")
        print(type(document))

        print("\nPage content:")
        print(document.page_content[:500])

        print("\nMetadata:")
        print(document.metadata)

    # --------------------------------
    # 2. Split documents
    # --------------------------------

    chunks = split_documents(
        documents,
        chunk_size=1000,
        chunk_overlap=200,
    )

    print("\n==============================")
    print("CHUNKED DOCUMENTS")
    print("==============================")

    print(f"Number of chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks[:5]):

        print(f"\n--- Chunk {i} ---")

        print("Type:",type(chunk))

        print("\nContent:")
        print(chunk.page_content[:500])

        print("\nMetadata:")
        print(chunk.metadata)

        print("\nCharacter count:")
        print(len(chunk.page_content))

    from collections import Counter

    page_counts = Counter(
        chunk.metadata["page"]
        for chunk in chunks
    )

    print("\nCHUNKS PER PAGE")

    for page, count in sorted(page_counts.items()):
        print(f"Page {page + 1}: {count} chunks")

    # --------------------------------
    # 3. Create embedding model
    # --------------------------------

    embedding_model = create_embedding_model()

    print("\n==============================")
    print("EMBEDDING")
    print("==============================")

    text = chunks[0].page_content

    vector = embedding_model.embed_query(text)

    print("Text:")
    print(text[:500])

    print("\nVector:")
    print(vector)

    print("\nVector type:")
    print(type(vector))

    print("\nVector dimensions:")
    print(len(vector))

if __name__ == "__main__":
    main()