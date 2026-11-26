# Research Insight Assistant

A conversational RAG (Retrieval-Augmented Generation) application that lets you upload research PDFs and ask questions about them. Built with LangChain, FastAPI, ChromaDB, and Streamlit.

## Overview

Research Insight Assistant is a tool for researchers and knowledge workers who need to quickly extract insights from collections of PDF documents. Upload your research papers, technical documents, or any PDF content, then ask natural language questions to get grounded answers with source citations.

## Features

- **PDF Upload & Indexing**: Upload PDFs through the UI or API; documents are automatically chunked, embedded, and indexed in ChromaDB
- **Conversational RAG**: Ask follow-up questions with full conversation history context
- **Source Citations**: Every answer includes references to specific documents and page numbers
- **Multiple Interfaces**: 
  - Web UI (Streamlit) for interactive use
  - REST API (FastAPI) for programmatic access
  - Terminal CLI for quick testing
- **Persistent Conversations**: Chat history stored in SQLite, survives restarts
- **Document Management**: List, upload, and delete indexed documents
- **Docker Support**: Multi-stage Dockerfile for production deployment

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│   Streamlit │────▶│    FastAPI  │────▶│   RAG Service   │
│   Frontend  │     │    Backend  │     │  (LangChain)    │
└─────────────┘     └─────────────┘     └────────┬────────┘
                                                  │
                    ┌─────────────────────────────┼─────────────────────────────┐
                    ▼                             ▼                             ▼
            ┌───────────────┐             ┌───────────────┐             ┌───────────────┐
            │  ChromaDB     │             │   SQLite      │             │   OpenRouter  │
            │  (Vector DB)  │             │  (Conversations)          │   (LLM API)   │
            └───────────────┘             └───────────────┘             └───────────────┘
```

### Core Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Vector Store** | ChromaDB | Semantic search over document chunks |
| **Embeddings** | sentence-transformers (via HuggingFace) | Convert text to vectors |
| **LLM** | OpenRouter (Claude Sonnet 4) | Generate answers |
| **Retriever** | LangChain + Chroma | Hybrid search (vector + keyword) |
| **Reranker** | Cross-encoder (disabled on free tier) | Re-rank retrieved chunks |
| **Conversation Store** | SQLite | Persistent chat history |
| **API** | FastAPI | REST endpoints |
| **Frontend** | Streamlit | Interactive web UI |

## Project Structure

```
research-insight-assistant/
├── api.py                      # FastAPI REST API
├── main.py                     # Terminal CLI entry point
├── streamlit_app.py            # Streamlit web UI
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Multi-stage Docker build
├── docker-compose.yml          # Local development stack
├── .env.example                # Environment variable template
├── app/
│   ├── rag/
│   │   ├── service.py          # Main RAG service orchestration
│   │   ├── chain.py            # RAG chain construction
│   │   └── conversation_store.py # SQLite conversation persistence
│   ├── retrieval/
│   │   ├── retriever.py        # Vector + keyword retrieval
│   │   └── reranker.py         # Cross-encoder reranking
│   ├── generation/
│   │   ├── llm.py              # LLM initialization (OpenRouter)
│   │   └── prompt.py           # Prompt templates
│   ├── ingestion/
│   │   ├── loaders.py          # PDF loading (pypdf)
│   │   ├── splitter.py         # Text chunking (RecursiveCharacterTextSplitter)
│   │   └── embedder.py         # Embedding model creation
│   ├── vectorstore/
│   │   └── store.py            # ChromaDB operations
│   └── evaluation/             # RAG evaluation scripts
└── data/
    ├── raw/                    # PDF storage (mounted as volume)
    └── chroma/                 # ChromaDB persistence (mounted as volume)
```

## Quick Start

### Prerequisites

- Python 3.12+
- OpenRouter API key (for LLM access)

### Local Development (Docker Compose)

```bash
# 1. Clone and configure
git clone <your-repo>
cd research-insight-assistant
cp .env.example .env
# Edit .env with your OPENROUTER_APIKEY

# 2. Start services
docker-compose up -d

# 3. Access
# Frontend: http://localhost:8501
# API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### Local Development (Manual)

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables
export OPENROUTER_APIKEY=your-key
export OPENROUTER_MODEL=anthropic/claude-sonnet-4

# 4. Start API (terminal 1)
python -m uvicorn api:app --reload

# 5. Start Streamlit (terminal 2)
streamlit run streamlit_app.py

# 6. Or use CLI (terminal 3)
python main.py
```

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENROUTER_APIKEY` | Yes | - | OpenRouter API key for LLM access |
| `OPENROUTER_MODEL` | No | `anthropic/claude-sonnet-4` | Model to use via OpenRouter |
| `API_URL` | No | `http://127.0.0.1:8000` | API endpoint (Streamlit reads this) |
| `DATA_DIR` | No | `data/raw` | Directory for PDF storage |
| `CHROMA_PERSIST_DIRECTORY` | No | `data/chroma` | ChromaDB persistence path |

### .env.example

```env
OPENROUTER_APIKEY=your-openrouter-key-here
OPENROUTER_MODEL=anthropic/claude-sonnet-4
```

## API Reference

### Base URL
```
http://localhost:8000
```

### Endpoints

#### Health Check
```
GET /health
```
Response: `{"status": "ok"}`

#### Ask a Question
```
POST /ask
Content-Type: application/json

{
  "question": "What are the main findings of the paper?",
  "session_id": "conversation-uuid"
}
```

Response:
```json
{
  "answer": "The paper finds that...",
  "sources": [
    {"file": "paper.pdf", "page": "5", "chunk_id": "abc123"},
    {"file": "paper.pdf", "page": "12", "chunk_id": "def456"}
  ]
}
```

#### Conversations
```
GET /conversations                    # List all conversations
POST /conversations                   # Create new conversation
GET /conversations/{id}/messages      # Get conversation history
```

#### Document Management
```
GET /documents                        # List indexed documents
POST /upload                          # Upload and index PDF (multipart/form-data)
DELETE /documents?filename=xxx.pdf    # Delete document and its chunks
```

## Usage Examples

### Using the Web UI

1. Open http://localhost:8501
2. Click "＋ New conversation" in sidebar
3. Upload PDFs using "Upload & Index" button
4. Ask questions in the chat input
5. View sources by expanding "View sources" under each answer

### Using the CLI

```bash
python main.py
```

```
Research Insight Assistant ready! Type 'exit' to stop.

YOU: What is the methodology used in the Smith et al. paper?

ASSISTANT: 
 The methodology combines...
 
Sources:
------------------------------
[Source 1] smith2023.pdf — Page 3
[Source 2] smith2023.pdf — Page 7
```

### Using the API (cURL)

```bash
# Create conversation
curl -X POST http://localhost:8000/conversations \
  -H "Content-Type: application/json" \
  -d '{"title": "Research Session"}'

# Ask question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarize the key findings", "session_id": "conv-uuid"}'

# Upload PDF
curl -X POST http://localhost:8000/upload \
  -F "file=@research_paper.pdf"

# List documents
curl http://localhost:8000/documents
```

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed guides on deploying to:

- **Railway** (Recommended - easiest)
- **Render**
- **Fly.io**
- **Google Cloud Run**
- **AWS ECS/Fargate**
- **DigitalOcean App Platform**

### Key Deployment Considerations

1. **Persistence**: Mount volumes for `data/raw` (PDFs) and `data/chroma` (vector DB)
2. **Secrets**: Use platform secret managers for `OPENROUTER_APIKEY`
3. **Scaling**: API is stateless; Streamlit needs sticky sessions
4. **Memory**: Reranker disabled by default for free tier deployments

## Evaluation

The project includes evaluation scripts in `app/evaluation/`:

```bash
# Run retrieval evaluation
python -m app.evaluation.retrieval_experiment

# Run answer quality evaluation
python -m app.evaluation.answer_eval
```

## Technical Details

### RAG Pipeline

1. **Ingestion**: PDFs → text extraction → chunking (1000 chars, 200 overlap) → embeddings → ChromaDB
2. **Retrieval**: Query → embedding → vector search (top 10) → optional reranking → top 5 chunks
3. **Generation**: Retrieved chunks + conversation history + question → LLM → grounded answer
4. **Citation**: Source metadata (filename, page) attached to each retrieved chunk

### Conversation Management

- Conversations stored in SQLite (`conversations.db`)
- In-memory LangChain history restored from SQLite on session start
- First user question becomes conversation title (truncated to 40 chars)

### Document Sync

- `sync_index()` runs on startup and after each upload
- Detects new, modified, and deleted PDFs via SHA256 hash comparison
- Incremental updates: only re-indexes changed files

## Troubleshooting

| Issue | Solution |
|-------|----------|
| ChromaDB permission errors | Ensure volume mounts have correct permissions (UID 1000) |
| Model download timeout | Increase container startup timeout |
| Out of memory | Use smaller models, disable reranker, increase container memory |
| Streamlit can't connect to API | Check `API_URL` env var, verify network connectivity |
| PDF upload fails | Check file size limits (20MB), verify volume mounts |

## Future Developments / Goals

- **Auth & Security**: User login (OAuth/JWT), roles, rate limiting, API keys
- **Document Support**: DOCX/TXT/MD/HTML/EPUB, OCR for scanned PDFs, auto metadata extraction
- **PostgreSQL**: Replace SQLite for multi-instance, conversation branching, export/import
- **Advanced RAG**: Multi-hop reasoning, table/figure understanding, hallucination detection, structured outputs
- **Observability**: Automated evaluation, LangSmith/LangFuse tracing, Prometheus/Grafana, A/B testing
- **Multi-Modal**: Image understanding (charts, diagrams), audio/video transcripts, cross-modal retrieval
- **Knowledge Graphs**: Entity extraction, relationship mapping, concept visualization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## Acknowledgments

- [LangChain](https://langchain.com/) for RAG orchestration
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [OpenRouter](https://openrouter.ai/) for LLM access
- [Streamlit](https://streamlit.io/) for the web UI
- [FastAPI](https://fastapi.tiangolo.com/) for the API framework

## 📜 License

This project is licensed under the **MIT License**.

---

## 👨‍💻 About the Author

### Prince Prem

Aspiring **AI Engineer** focused on building practical AI systems and understanding the engineering foundations behind modern RAG and Agentic AI applications.

**GitHub:**

`https://github.com/Princeprem117`

**LinkedIn:**

`https://www.linkedin.com/in/princeprem14/`