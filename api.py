from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from app.rag.service import RAGService

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build and synchronize the RAG pipeline once at startup.
    app.state.rag_service = RAGService()
    yield
    # Add resource cleanup here if needed later.

app = FastAPI(
    title="Research Insight Assistant API",
    description="Ask questions about your indexed research PDFs.",
    version="1.0.0",
    lifespan=lifespan,
)

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(
        default="default",
        min_length=1,
        max_length=100,
    )


class Source(BaseModel):
    file: str
    page: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest, request: Request):
    question = payload.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be blank.",
        )

    try:
        result = request.app.state.rag_service.ask(
            question=question,
            session_id=payload.session_id,
        )
    except Exception as exc:
        print(f"RAG request failed: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate an answer.",
        ) from exc

    sources = []
    seen = set()

    for doc in result.get("documents", []):
        file_name = Path(
            doc.metadata.get(
                "source",
                doc.metadata.get("source_path", "Unknown"),
            )
        ).name

        page = str(
            doc.metadata.get(
                "page_label",
                doc.metadata.get("page", "Unknown"),
            )
        )

        key = (file_name, page)
        if key not in seen:
            seen.add(key)
            sources.append(
                Source(file=file_name, page=page)
            )

    return AskResponse(
        answer=result["answer"],
        sources=sources,
    )
