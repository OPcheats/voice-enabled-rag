"""
Central RAGOrchestrator — Tech Doc §12.

Pipeline stages (from Tech Doc §12):
  1. validate_input      → guardrails §11.1
  2. transcribe          → stt.py
  3. validate_query      → guardrails §11.2 + §11.4
  4. retrieve            → retrieval.py
  5. validate_context    → guardrails §11.3
  6. generate            → generation.py
  7. verify_grounding    → guardrails §11.5
  8. format_response     → returns AskResponse

The orchestrator NEVER raises — it always returns AskResponse with
an appropriate status and a user-safe message (Tech Doc §14).

Latency is measured per stage using time.perf_counter() (Tech Doc §16).
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

from app.guardrails import (
    Guardrails,
    InputValidationError,
    OffTopicError,
    UnsafeInputError,
)
from app.generation import AnswerGenerator, GenerationError
from app.retrieval import RetrievalInterface, RetrievalError
from app.schemas import AskResponse, RetrievedChunk, StageLatencies
from app.stt import SarvamSTTService, STTError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# User-facing refusal messages — PRD §8
# ---------------------------------------------------------------------------
_MSG_INSUFFICIENT_CONTEXT = (
    "I couldn't find enough relevant information in the provided knowledge base "
    "to answer this question."
)
_MSG_OFF_TOPIC = "I can only answer questions supported by the provided knowledge base."
_MSG_UNSAFE = "I can't assist with that request."
_MSG_SERVICE_ERROR = "The service is temporarily unavailable. Please try again."
_MSG_NOT_GROUNDED = (
    "I was unable to produce a sufficiently grounded answer from the available information."
)


def _ms(elapsed: float) -> float:
    """Convert perf_counter() seconds delta to milliseconds."""
    return round(elapsed * 1000, 2)


class RAGOrchestrator:
    """Orchestrates the complete Voice-Enabled RAG pipeline.

    Usage::

        orchestrator = RAGOrchestrator()
        response = await orchestrator.run(audio_bytes=b"...")
    """

    def __init__(
        self,
        stt: Optional[SarvamSTTService] = None,
        retriever: Optional[RetrievalInterface] = None,
        generator: Optional[AnswerGenerator] = None,
        guardrails: Optional[Guardrails] = None,
    ) -> None:
        self.stt = stt or SarvamSTTService()
        self.retriever = retriever or RetrievalInterface()
        self.guardrails = guardrails or Guardrails()
        # AnswerGenerator reads from env; may raise ValueError if LLM_API_KEY missing.
        # We defer construction so the app can start even without an LLM key configured
        # and return a controlled error only when /ask is actually called.
        self._generator = generator
        self._generator_init_error: Optional[str] = None
        if self._generator is None:
            try:
                self._generator = AnswerGenerator()
            except ValueError as exc:
                self._generator_init_error = str(exc)
                logger.warning("AnswerGenerator not initialized: %s", exc)

    async def run(
        self,
        audio_bytes: bytes,
        audio_filename: str = "audio.wav",
        content_type: Optional[str] = None,
    ) -> AskResponse:
        """Execute the full RAG pipeline and return a structured AskResponse.

        Never raises — all errors are converted to controlled AskResponse objects.
        """
        pipeline_start = time.perf_counter()
        latencies = StageLatencies()
        transcript = ""
        chunks: List[RetrievedChunk] = []

        # -------------------------------------------------------------------
        # Stage 1+2: Transcribe audio
        # -------------------------------------------------------------------
        try:
            t0 = time.perf_counter()
            transcript = await self.stt.transcribe(
                audio_bytes=audio_bytes,
                filename=audio_filename,
                content_type=content_type,
            )
            latencies.stt_ms = _ms(time.perf_counter() - t0)
            logger.info("STT complete: %r", transcript[:80])
        except STTError as exc:
            logger.error("STT failed: %s", exc)
            latencies.total_ms = _ms(time.perf_counter() - pipeline_start)
            return self._error_response(latencies, transcript="")

        # -------------------------------------------------------------------
        # Stage 3a: Input validation (§11.1)
        # -------------------------------------------------------------------
        t0 = time.perf_counter()
        try:
            self.guardrails.validate_input(transcript)
        except InputValidationError as exc:
            latencies.guardrail_ms += _ms(time.perf_counter() - t0)
            latencies.total_ms = _ms(time.perf_counter() - pipeline_start)
            return self._refused_response(
                str(exc), transcript=transcript, latencies=latencies
            )

        # Stage 3b: Unsafe input check (§11.4)
        try:
            self.guardrails.check_unsafe(transcript)
        except UnsafeInputError as exc:
            latencies.guardrail_ms += _ms(time.perf_counter() - t0)
            latencies.total_ms = _ms(time.perf_counter() - pipeline_start)
            return self._unsafe_response(transcript=transcript, latencies=latencies)

        # Stage 3c: Off-topic check (§11.2) — before expensive retrieval
        try:
            self.guardrails.check_off_topic(transcript)
        except OffTopicError as exc:
            latencies.guardrail_ms += _ms(time.perf_counter() - t0)
            latencies.total_ms = _ms(time.perf_counter() - pipeline_start)
            return self._refused_response(
                _MSG_OFF_TOPIC, transcript=transcript, latencies=latencies
            )
        latencies.guardrail_ms += _ms(time.perf_counter() - t0)

        # -------------------------------------------------------------------
        # Stage 4: Retrieve context
        # -------------------------------------------------------------------
        try:
            t0 = time.perf_counter()
            chunks = await self.retriever.retrieve(transcript)
            latencies.retrieval_ms = _ms(time.perf_counter() - t0)
            logger.info("Retrieved %d chunks", len(chunks))
        except RetrievalError as exc:
            logger.error("Retrieval failed: %s", exc)
            latencies.total_ms = _ms(time.perf_counter() - pipeline_start)
            return self._error_response(latencies, transcript=transcript)

        # -------------------------------------------------------------------
        # Stage 5: Context validation (§11.3)
        # -------------------------------------------------------------------
        t0 = time.perf_counter()
        if not self.guardrails.is_context_sufficient(chunks):
            latencies.guardrail_ms += _ms(time.perf_counter() - t0)
            latencies.total_ms = _ms(time.perf_counter() - pipeline_start)
            return self._refused_response(
                _MSG_INSUFFICIENT_CONTEXT,
                transcript=transcript,
                latencies=latencies,
                status="refused",
            )
        latencies.guardrail_ms += _ms(time.perf_counter() - t0)

        # -------------------------------------------------------------------
        # Stage 6: Generate answer
        # -------------------------------------------------------------------
        if self._generator is None:
            # LLM not configured
            err_msg = self._generator_init_error or "LLM not configured."
            logger.error("Generation skipped: %s", err_msg)
            latencies.total_ms = _ms(time.perf_counter() - pipeline_start)
            return self._error_response(latencies, transcript=transcript)

        try:
            t0 = time.perf_counter()
            answer = await self._generator.generate(transcript, chunks)
            latencies.generation_ms = _ms(time.perf_counter() - t0)
            logger.info("Generation complete: %r", answer[:80])
        except GenerationError as exc:
            logger.error("Generation failed: %s", exc)
            latencies.total_ms = _ms(time.perf_counter() - pipeline_start)
            return self._error_response(latencies, transcript=transcript)

        # -------------------------------------------------------------------
        # Stage 7: Grounding verification (§11.5)
        # -------------------------------------------------------------------
        t0 = time.perf_counter()
        grounded = self.guardrails.verify_grounding(answer, chunks)
        latencies.guardrail_ms += _ms(time.perf_counter() - t0)

        if not grounded:
            latencies.total_ms = _ms(time.perf_counter() - pipeline_start)
            return AskResponse(
                answer=_MSG_NOT_GROUNDED,
                transcript=transcript,
                sources=chunks,
                grounded=False,
                status="refused",
                latency_ms=latencies.total_ms,
                stage_latencies=latencies,
            )

        # -------------------------------------------------------------------
        # Stage 8: Format and return successful response
        # -------------------------------------------------------------------
        latencies.total_ms = _ms(time.perf_counter() - pipeline_start)
        logger.info("Pipeline complete in %.1f ms", latencies.total_ms)

        return AskResponse(
            answer=answer,
            transcript=transcript,
            sources=chunks,
            grounded=True,
            status="success",
            latency_ms=latencies.total_ms,
            stage_latencies=latencies,
        )

    async def run_text(self, text_query: str) -> AskResponse:
        """Helper to run text-only queries bypassing STT stage."""
        pipeline_start = time.perf_counter()
        latencies = StageLatencies()
        transcript = text_query.strip()
        chunks: List[RetrievedChunk] = []

        # Stage 3a: Input validation
        t0 = time.perf_counter()
        try:
            self.guardrails.validate_input(transcript)
        except InputValidationError as exc:
            latencies.guardrail_ms += _ms(time.perf_counter() - t0)
            latencies.total_ms = _ms(time.perf_counter() - pipeline_start)
            return self._refused_response(str(exc), transcript=transcript, latencies=latencies)

        # Stage 3b: Unsafe check
        try:
            self.guardrails.check_unsafe(transcript)
        except UnsafeInputError:
            latencies.guardrail_ms += _ms(time.perf_counter() - t0)
            latencies.total_ms = _ms(time.perf_counter() - pipeline_start)
            return self._unsafe_response(transcript=transcript, latencies=latencies)

        # Stage 3c: Off-topic check
        try:
            self.guardrails.check_off_topic(transcript)
        except OffTopicError:
            latencies.guardrail_ms += _ms(time.perf_counter() - t0)
            latencies.total_ms = _ms(time.perf_counter() - pipeline_start)
            return self._refused_response(_MSG_OFF_TOPIC, transcript=transcript, latencies=latencies)
        latencies.guardrail_ms += _ms(time.perf_counter() - t0)

        # Stage 4: Retrieval
        try:
            t0 = time.perf_counter()
            chunks = await self.retriever.retrieve(transcript)
            latencies.retrieval_ms = _ms(time.perf_counter() - t0)
        except RetrievalError:
            latencies.total_ms = _ms(time.perf_counter() - pipeline_start)
            return self._error_response(latencies, transcript=transcript)

        # Stage 5: Context validation
        t0 = time.perf_counter()
        if not self.guardrails.is_context_sufficient(chunks):
            latencies.guardrail_ms += _ms(time.perf_counter() - t0)
            latencies.total_ms = _ms(time.perf_counter() - pipeline_start)
            return self._refused_response(_MSG_INSUFFICIENT_CONTEXT, transcript=transcript, latencies=latencies)
        latencies.guardrail_ms += _ms(time.perf_counter() - t0)

        # Stage 6: Generation
        if self._generator is None:
            latencies.total_ms = _ms(time.perf_counter() - pipeline_start)
            return self._error_response(latencies, transcript=transcript)

        try:
            t0 = time.perf_counter()
            answer = await self._generator.generate(transcript, chunks)
            latencies.generation_ms = _ms(time.perf_counter() - t0)
        except GenerationError:
            latencies.total_ms = _ms(time.perf_counter() - pipeline_start)
            return self._error_response(latencies, transcript=transcript)

        # Stage 7: Grounding
        t0 = time.perf_counter()
        grounded = self.guardrails.verify_grounding(answer, chunks)
        latencies.guardrail_ms += _ms(time.perf_counter() - t0)

        latencies.total_ms = _ms(time.perf_counter() - pipeline_start)
        return AskResponse(
            answer=answer if grounded else _MSG_NOT_GROUNDED,
            transcript=transcript,
            sources=chunks,
            grounded=grounded,
            status="success" if grounded else "refused",
            latency_ms=latencies.total_ms,
            stage_latencies=latencies,
        )


    # -----------------------------------------------------------------------
    # Response helpers — PRD §8 user-facing states
    # -----------------------------------------------------------------------
    @staticmethod
    def _error_response(latencies: StageLatencies, transcript: str = "") -> AskResponse:
        return AskResponse(
            answer=_MSG_SERVICE_ERROR,
            transcript=transcript,
            sources=[],
            grounded=False,
            status="error",
            latency_ms=latencies.total_ms,
            stage_latencies=latencies,
        )

    @staticmethod
    def _refused_response(
        message: str,
        transcript: str,
        latencies: StageLatencies,
        status: str = "refused",
    ) -> AskResponse:
        return AskResponse(
            answer=message,
            transcript=transcript,
            sources=[],
            grounded=False,
            status=status,  # type: ignore[arg-type]
            latency_ms=latencies.total_ms,
            stage_latencies=latencies,
        )

    @staticmethod
    def _unsafe_response(transcript: str, latencies: StageLatencies) -> AskResponse:
        return AskResponse(
            answer=_MSG_UNSAFE,
            transcript=transcript,
            sources=[],
            grounded=False,
            status="unsafe",
            latency_ms=latencies.total_ms,
            stage_latencies=latencies,
        )
