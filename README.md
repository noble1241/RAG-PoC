# RAG Service

Production-grade Retrieval-Augmented Generation service.

## Stack

- **Backend**: FastAPI (Python 3.11+), async OpenAI SDK, ChromaDB HTTP client, tiktoken, tenacity
- **Frontend**: React + TypeScript + Vite, Server-Sent Events streaming
- **Vector DB**: ChromaDB in server mode

## Quick Start (local dev — no Docker needed)

**Terminal 1 — ChromaDB**
```bash
cd backend
pip install -r requirements.txt
chroma run --path ./chroma_data --port 8000
```
Data is saved to `backend/chroma_data/` and persists between restarts.

**Terminal 2 — Backend**
```bash
cd backend
python run.py
```

**Terminal 3 — Frontend**
```bash
cd frontend
npm install
npm run dev
```

- Backend: http://localhost:8080
- Frontend: http://localhost:5173
- ChromaDB: http://localhost:8000

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /health | Liveness check |
| GET | /ready | Readiness check (verifies Chroma) |
| POST | /documents | Ingest text `{"text":"...", "source":"name"}` |
| POST | /documents/upload | Ingest file (.pdf / .docx / .xlsx / .xls / .pptx / .csv / .txt / .md — converted to Markdown via MarkItDown) |
| POST | /documents/extract-policy | Extract structured PolicyDocument(s) from a policy file → JSON artifacts + RAG ingest (SSE not used). Optional `?sheet=&policy=` |
| POST | /chat | Chat query `{"query":"..."}` → SSE stream |

### POST /chat response (SSE)

Three event types streamed in order:

```
data: {"type":"sources","sources":[{"source":"doc.txt","chunk_index":0,"content":"...","score":0.92}]}

data: {"type":"token","content":"Hello"}
data: {"type":"token","content":" world"}

data: {"type":"done"}
```

### POST /documents response

```json
{
  "document_id": "a1b2c3d4e5f6g7h8",
  "source": "my-doc.txt",
  "chunk_count": 12,
  "tokens_processed": 5847
}
```

## Configuration

All config via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| OPENAI_API_KEY | required | OpenAI API key |
| CHROMA_HOST | localhost | ChromaDB host |
| CHROMA_PORT | 8000 | ChromaDB port |
| EMBEDDING_MODEL | text-embedding-3-small | Embedding model |
| CHAT_MODEL | gpt-4o-mini | Chat completion model |
| TOP_K | 4 | Retrieved chunks per query |
| CHUNK_SIZE | 500 | Tokens per chunk |
| CHUNK_OVERLAP | 50 | Token overlap between chunks |
| CONVERSION_TIMEOUT_SECONDS | 30 | Max seconds for MarkItDown to convert one upload |
| SAVE_CONVERTED_MARKDOWN | true | Dump each upload's converted Markdown to disk for inspection |
| CONVERTED_OUTPUT_DIR | converted_output | Where converted `<filename>.md` files are written (relative to `backend/`) |
| ALLOWED_ORIGINS | ["http://localhost:5173"] | CORS allowed origins |
| LOG_LEVEL | INFO | Logging level |

## Tests

```bash
cd backend
pytest
```

