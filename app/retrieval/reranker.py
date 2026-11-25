"""Cross-encoder reranker for document re-ranking using sentence-transformers."""

from sentence_transformers import CrossEncoder


def create_reranker(
    model_name: str = "BAAI/bge-reranker-v2-m3",
    top_n: int = 3,
):
    """Create a cross-encoder reranker for document re-ranking.

    Args:
        model_name: Name of the cross-encoder model from sentence-transformers.
        top_n: Number of top documents to return after re-ranking.

    Returns:
        A CrossEncoder model ready for re-ranking.
    """
    cross_encoder = CrossEncoder(model_name)

    def rerank(query: str, documents: list, top_n: int = top_n) -> list:
        """Re-rank documents by relevance to the query.

        Args:
            query: The user's search query.
            documents: List of Document objects to re-rank.
            top_n: Number of top documents to return.

        Returns:
            Re-ordered list of Document objects, limited to top_n.
        """
        # Prepare pairs of (query, document_content)
        pairs = [(query, doc.page_content) for doc in documents]

        # Get relevance scores from the cross-encoder
        scores = cross_encoder.predict(pairs)

        # Sort documents by score (descending)
        sorted_docs = [doc for _, doc in sorted(zip(scores, documents), reverse=True)]

        # Return top_n documents
        return sorted_docs[:top_n]

    return rerank