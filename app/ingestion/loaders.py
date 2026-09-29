from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from pathlib import Path

def load_pdf(file_path: str) -> list[Document]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    loader = PyPDFLoader(path)
    documents = loader.load()

    return documents