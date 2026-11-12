"""FastAPI app. Run locally:  uvicorn backend.app:app --reload
Railway runs the Procfile:    uvicorn backend.app:app --host 0.0.0.0 --port $PORT"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import config
from .evaluate import run_evaluation
from .rag import RAGPipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = RAGPipeline()  # ingest + chunk + embed once at start-up (~1 s for the bundled data)
    yield


app = FastAPI(title="Energy Operations Document RAG Assistant", version="1.0.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    top_k: int = Field(default=config.TOP_K, ge=1, le=8)


@app.get("/health")
def health():
    p: RAGPipeline = app.state.pipeline
    return {"status": "ok", "documents": len(p.docs), "chunks": len(p.chunks),
            "generator": "llm+fallback" if config.LLM_PROVIDER == "anthropic" and config.ANTHROPIC_API_KEY else "extractive"}


@app.post("/api/ask")
def ask(req: AskRequest):
    return app.state.pipeline.ask(req.question.strip(), req.top_k)


@app.get("/api/documents")
def documents():
    return app.state.pipeline.list_documents()


@app.post("/api/documents")
async def upload(file: UploadFile = File(...)):
    """Add an SOP (.md/.txt). Stored in memory only: it is lost on restart (see README, limitations)."""
    if not (file.filename or "").lower().endswith((".md", ".txt")):
        raise HTTPException(400, "Only .md or .txt files are supported")
    raw = await file.read(config.MAX_UPLOAD_BYTES + 1)
    if len(raw) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File larger than {config.MAX_UPLOAD_BYTES // 1024} KB")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(400, "File must be UTF-8 text")
    if len(text.split()) < 20:
        raise HTTPException(400, "Document is too short to index")
    doc = app.state.pipeline.add_document(text, file.filename)
    return {"doc_id": doc.doc_id, "title": doc.title, "documents": len(app.state.pipeline.docs)}


@app.get("/api/evaluate")
def evaluate():
    return run_evaluation(app.state.pipeline, with_ablation=False)


@app.get("/")
def index():
    return FileResponse(config.FRONTEND_DIR / "index.html")
