import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request , File, UploadFile
from pydantic import BaseModel, Field

from app.rag.service import RAGService

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag_service = RAGService()
    yield


app = FastAPI(
    title="Research Insight Assistant API",
    description="Ask questions about your indexed research PDFs.",
    version="1.0.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(min_length=1, max_length=100)


class CreateConversationRequest(BaseModel):
    title: str = Field(default="New conversation", max_length=100)


class Source(BaseModel):
    file: str
    page: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/conversations")
def list_conversations(request: Request):
    return request.app.state.rag_service.list_conversations()


@app.post("/conversations")
def create_conversation(
    payload: CreateConversationRequest,
    request: Request,
):
    service = request.app.state.rag_service
    conversation_id = str(uuid.uuid4())
    service.create_conversation(conversation_id)

    if payload.title != "New conversation":
        service.conversation_store.update_title(
            conversation_id, payload.title
        )

    return {
        "id": conversation_id,
        "title": payload.title,
    }


@app.get("/conversations/{conversation_id}/messages")
def get_conversation_messages(
    conversation_id: str,
    request: Request,
):
    service = request.app.state.rag_service

    if not service.conversation_store.get_conversation(conversation_id):
        raise HTTPException(
            status_code=404,
            detail="Conversation not found",
        )

    messages = service.get_conversation_messages(conversation_id)

    for message in messages:
        message["sources"] = json.loads(
            message.get("sources") or "[]"
        )

    return messages


@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest, request: Request):
    question = payload.question.strip()
    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be blank.",
        )

    service = request.app.state.rag_service

    if not service.conversation_store.get_conversation(
        payload.session_id
    ):
        raise HTTPException(
            status_code=404,
            detail="Conversation not found. Create it first.",
        )

    try:
        result = service.ask(
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

        if (file_name, page) not in seen:
            seen.add((file_name, page))
            sources.append(Source(file=file_name, page=page))

    return AskResponse(
        answer=result["answer"],
        sources=sources,
    )


MAX_PDF_SIZE = 20 * 1024 * 1024  # 20 MB


@app.post("/upload")
async def upload_pdf(
    request: Request,
    file: UploadFile = File(...),
):
    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed.",
        )

    content = bytearray()

    try:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break

            content.extend(chunk)

            if len(content) > MAX_PDF_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail="PDF must be 20 MB or smaller.",
                )

        if not content.startswith(b"%PDF-"):
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is not a valid PDF.",
            )

        service = request.app.state.rag_service
        upload_dir = Path(service.data_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Keep existing files safe by generating a unique filename.
        original_name = Path(file.filename).name
        destination = upload_dir / original_name

        if destination.exists():
            destination = upload_dir / (
                f"{destination.stem}_{uuid.uuid4().hex[:8]}.pdf"
            )

        destination.write_bytes(content)

        try:
            service.sync_index()
        except Exception as exc:
            print(f"PDF indexing failed: {exc}")
            raise HTTPException(
                status_code=500,
                detail="Upload succeeded, but PDF indexing failed.",
            ) from exc

        return {
            "message": "PDF uploaded and indexed successfully.",
            "filename": destination.name,
        }

    finally:
        await file.close()


@app.get("/documents")
def list_documents(request: Request):
    service = request.app.state.rag_service
    return service.list_documents()


@app.delete("/documents")
def delete_document(filename: str, request: Request):
    service = request.app.state.rag_service

    try:
        return service.delete_document(filename)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        print(f"Document deletion failed: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Failed to delete the document and synchronize the index.",
        ) from exc
