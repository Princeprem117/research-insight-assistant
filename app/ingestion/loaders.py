from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
)
from langchain_core.documents import Document


def load_pdf_file(file_path: str | Path) -> list[Document]:
    """Load pages from one PDF file."""
    pdf_path = Path(file_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    if not pdf_path.is_file():
        raise ValueError(
            f"Expected a PDF file: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Expected a .pdf file: {pdf_path}"
        )

    return PyPDFLoader(str(pdf_path)).load()


def load_pdf(directory_path: str) -> list[Document]:
    """Load all PDFs in a directory."""
    directory = Path(directory_path)

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory not found: {directory_path}"
        )

    if not directory.is_dir():
        raise NotADirectoryError(
            f"Not a directory: {directory_path}"
        )

    loader = DirectoryLoader(
        path=str(directory),
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        silent_errors=False,
    )

    documents = loader.load()

    if not documents:
        raise ValueError(
            f"No documents found in directory: {directory_path}"
        )

    return documents