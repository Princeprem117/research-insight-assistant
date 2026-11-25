from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter


def split_documents(
    documents: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    use_tokens: bool = True,
) -> list[Document]:
    """Split documents into chunks.

    Args:
        documents: List of documents to split.
        chunk_size: Maximum chunk size (characters if use_tokens=False, tokens if True).
        chunk_overlap: Overlap between chunks (characters or tokens).
        use_tokens: If True, use TokenTextSplitter for token-aware splitting.

    Returns:
        List of document chunks.
    """
    if use_tokens:
        # Use token-aware splitting with tiktoken (compatible with OpenAI models)
        # Default to cl100k_base encoding (used by GPT-4, GPT-3.5, etc.)
        splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name="cl100k_base",
        )
    else:
        # Fallback to character-based splitting
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    chunks = splitter.split_documents(documents)

    return chunks