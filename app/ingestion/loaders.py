from langchain_community.document_loaders import PyPDFLoader , DirectoryLoader
from langchain_core.documents import Document

from pathlib import Path

def load_pdf(directory_path: str) -> list[Document]:
    directory = Path(directory_path)
    if not directory.exists():
        raise FileNotFoundError(f"directory not found: {directory_path}")

    if not directory.is_dir():
        raise NotADirectoryError(f"not a directory: {directory_path}")

    loader = DirectoryLoader(
        path = str(directory),
        glob = "*.pdf",
        loader_cls=PyPDFLoader,
        show_progress = True,
        silent_errors=False,
    )


    documents = loader.load()

    if not documents:
        raise ValueError(f"no documents found in directory: {directory_path}")

    return documents