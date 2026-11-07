import logging
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.retriever import MAX_QUESTION_CHARS
from app.backend.rag_service import RagService

logger = logging.getLogger(__name__)

app = FastAPI(title="Document AI Assistant API", version="1.0.0")


@lru_cache(maxsize=1)
def get_service() -> RagService:
    return RagService()


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


class SourceOut(BaseModel):
    source: str
    label: str
    score: float
    text: str


class AskResponse(BaseModel):
    question: str
    status: str
    answer: str
    validated: bool
    attempts: int
    sources: list[SourceOut]
    trace: list[str]


class UploadResponse(BaseModel):
    chunks: dict[str, int]
    errors: list[str]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "app": "document-ai-assistant", "llm_configured": bool(settings.groq_api_key)}


@app.get("/documents")
def documents(service: RagService = Depends(get_service)) -> dict:
    return {"documents": service.store.list_sources(), "chunks": service.store.count()}


@app.post("/upload", response_model=UploadResponse)
async def upload(files: list[UploadFile], service: RagService = Depends(get_service)) -> UploadResponse:
    if len(files) > settings.max_files:
        raise HTTPException(400, f"Too many files: maximum is {settings.max_files}.")
    limit = settings.max_file_mb * 1024 * 1024
    payload = []
    for f in files:
        # Read one byte past the limit so oversized files are rejected without loading them fully.
        data = await f.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(413, f"{f.filename}: File exceeds the {settings.max_file_mb} MB limit.")
        payload.append((f.filename or "unnamed", data))
    try:
        report = service.ingest(payload)
    except Exception:
        logger.exception("Indexing failed")
        raise HTTPException(500, "Indexing failed due to an internal error.") from None
    if not report.chunk_counts:
        raise HTTPException(422, {"message": "No documents were indexed.", "errors": report.errors})
    return UploadResponse(chunks=report.chunk_counts, errors=report.errors)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, service: RagService = Depends(get_service)) -> AskResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(400, "Please enter a question.")
    try:
        result = service.ask(question)
    except Exception:
        logger.exception("Question answering failed")
        raise HTTPException(500, "Something went wrong while answering.") from None
    return AskResponse(
        question=question,
        status=result.status,
        answer=result.answer,
        validated=result.validated,
        attempts=result.attempts,
        sources=[
            SourceOut(source=s.metadata.get("source", ""), label=s.source_label, score=s.score, text=s.text)
            for s in result.sources
        ],
        trace=result.trace,
    )


@app.post("/reset")
def reset(service: RagService = Depends(get_service)) -> dict:
    service.store.reset()
    return {"status": "ok", "message": "Knowledge base cleared."}
