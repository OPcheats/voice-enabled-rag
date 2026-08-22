"""
Pydantic data contracts for Voice-Enabled RAG — Member 2.

Defined by Tech Doc §7 and PRD FR-11.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Status literals — Tech Doc §7
# ---------------------------------------------------------------------------
Status = Literal["success", "refused", "unsafe", "error"]


# ---------------------------------------------------------------------------
# Retrieved chunk — Tech Doc §7
# ---------------------------------------------------------------------------
class RetrievedChunk(BaseModel):
    """A single chunk returned by Member 1's retrieval system."""

    chunk_id: str = Field(..., description="Unique identifier for this chunk")
    document_id: str = Field(..., description="ID of the source document")
    text: str = Field(..., description="Raw text content of the chunk")
    score: float = Field(..., description="Relevance score (higher = more relevant)")
    source: Optional[str] = Field(None, description="Human-readable source reference")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Any additional metadata from Member 1")


# ---------------------------------------------------------------------------
# Stage-level latency — Tech Doc §16
# ---------------------------------------------------------------------------
class StageLatencies(BaseModel):
    """Breakdown of measured latencies per pipeline stage (milliseconds)."""

    stt_ms: float = 0.0
    embedding_ms: float = 0.0
    retrieval_ms: float = 0.0
    reranking_ms: float = 0.0
    generation_ms: float = 0.0
    guardrail_ms: float = 0.0
    total_ms: float = 0.0


# ---------------------------------------------------------------------------
# API response — Tech Doc §7, PRD FR-11
# ---------------------------------------------------------------------------
class AskResponse(BaseModel):
    """Structured response returned by POST /ask."""

    answer: str = Field(..., description="Generated answer text or refusal message")
    transcript: str = Field("", description="Speech-to-text transcript of the voice input")
    sources: List[RetrievedChunk] = Field(default_factory=list, description="Retrieved chunks used to produce the answer")
    grounded: bool = Field(..., description="True if the answer is supported by retrieved context")
    status: Status = Field(..., description="Pipeline outcome: success | refused | unsafe | error")
    latency_ms: float = Field(..., description="Total end-to-end latency in milliseconds")
    stage_latencies: StageLatencies = Field(default_factory=StageLatencies, description="Per-stage breakdown")


# ---------------------------------------------------------------------------
# Health check response — PRD §9
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
