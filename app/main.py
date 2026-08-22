"""
FastAPI application entry point — Tech Doc §15, PRD §9.

Endpoints:
  POST /ask    — process voice audio through full RAG pipeline
  GET  /health — liveness check

Input for POST /ask:
  multipart/form-data with a single 'audio' field (audio file bytes).

Output for POST /ask:
  AskResponse JSON (answer, transcript, sources, grounded, status, latency_ms, stage_latencies)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.pipeline import RAGOrchestrator
from app.schemas import AskResponse, HealthResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application lifespan — build orchestrator once at startup
# ---------------------------------------------------------------------------
_orchestrator: RAGOrchestrator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _orchestrator
    logger.info("Starting Voice-Enabled RAG API...")
    _orchestrator = RAGOrchestrator()
    logger.info("RAGOrchestrator initialized.")
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Voice-Enabled RAG API",
    description=(
        "Member 2 — Voice / AI / Application layer for HH Goa 2026. "
        "Converts voice audio to a grounded answer using Sarvam STT + LLM + FAISS retrieval."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow UI on any origin during development
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Exception handler — never expose secrets or stack traces (Tech Doc §14)
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again."},
    )


UI_DIR = Path(__file__).parent.parent / "ui"


# ---------------------------------------------------------------------------
# Web UI route — serve ui/index.html at http://localhost:8000/
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def serve_ui():
    index_file = UI_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse({"message": "Voice-Enabled RAG API active"})


# ---------------------------------------------------------------------------
# GET /health — PRD §9
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Liveness check. Returns {"status": "ok"} when the server is running."""
    return HealthResponse()


# ---------------------------------------------------------------------------
# POST /ask — Tech Doc §15, PRD §9
# ---------------------------------------------------------------------------
@app.post("/ask", response_model=AskResponse, tags=["RAG"])
async def ask(
    audio: UploadFile = File(..., description="Audio recording of the user's question"),
) -> AskResponse:
    """
    Process a voice question through the complete RAG pipeline.

    1. Read audio bytes from the uploaded file.
    2. Pass through RAGOrchestrator (STT → validation → retrieval → generation → grounding).
    3. Return AskResponse with answer, transcript, sources, grounded status, and latency.
    """
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="Service not ready. Try again shortly.")

    audio_bytes = await audio.read()
    if not audio_bytes or len(audio_bytes) == 0:
        logger.warning("Received empty audio file (0 bytes).")
        raise HTTPException(status_code=400, detail="Audio file is empty.")

    filename = audio.filename or "audio.wav"
    content_type = audio.content_type or "audio/wav"

    logger.info(
        "Received audio: filename=%s, content_type=%s, size_bytes=%d",
        filename,
        content_type,
        len(audio_bytes),
    )

    response = await _orchestrator.run(
        audio_bytes=audio_bytes,
        audio_filename=filename,
        content_type=content_type,
    )
    return response
