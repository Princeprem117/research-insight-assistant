import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
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
