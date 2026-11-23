from langchain_core.documents import Document
from langchain_community.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


def create_reranker(
    model_name: str = "BAAI/bge-reranker-v2-m3",
    top_n: int = 3,
) -> CrossEncoderReranker:
    """Create a cross-encoder reranker for document re-ranking."""
    cross_encoder = HuggingFaceCrossEncoder(model_name=model_name)
    reranker = CrossEncoderReranker(
        model=cross_encoder,
        top_n=top_n,
    )
    return reranker