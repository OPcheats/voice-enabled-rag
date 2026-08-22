"""
Guardrails — Tech Doc §11.

Five independent checks:
  §11.1 Input Validation    — empty/malformed/too-long transcript
  §11.2 Off-Topic Detection — query unrelated to knowledge base
  §11.3 Low Retrieval Confidence — retrieved scores below threshold
  §11.4 Unsafe Input        — inappropriate or harmful requests
  §11.5 Grounding Verification — answer unsupported by context

Each check raises a specific exception or returns a bool so the orchestrator
(pipeline.py) can build the correct AskResponse without the guardrails knowing
about response formatting.

Environment variables:
  MIN_RETRIEVAL_SCORE — configurable threshold (default 0.3, Tech Doc §11.3)
"""

from __future__ import annotations

import logging
import os
import re
from typing import List

from dotenv import load_dotenv

from app.schemas import RetrievedChunk

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configurable limits (Tech Doc §11.1)
# ---------------------------------------------------------------------------
MAX_QUERY_LENGTH = 2000  # characters

# ---------------------------------------------------------------------------
# Unsafe input patterns (Tech Doc §11.4)
# Basic keyword-level check — expand as needed during integration.
# ---------------------------------------------------------------------------
_UNSAFE_PATTERNS = re.compile(
    r"\b(ignore (previous|all) instructions?|jailbreak|bypass|system prompt|"
    r"you are now|disregard your|act as|roleplay as|pretend you are)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Guardrail exceptions
# ---------------------------------------------------------------------------

class InputValidationError(Exception):
    """Raised when the transcript or query fails validation (§11.1)."""


class UnsafeInputError(Exception):
    """Raised when the query matches unsafe content patterns (§11.4)."""


class OffTopicError(Exception):
    """Raised when the query appears off-topic for the knowledge base (§11.2)."""


class Guardrails:
    """Encapsulates all guardrail checks for the RAG pipeline.

    Usage::

        rails = Guardrails()

        # 1. validate input
        rails.validate_input(transcript)

        # 2. off-topic check
        rails.check_off_topic(query)

        # 3. low retrieval confidence
        if not rails.is_context_sufficient(chunks):
            return refused_response()

        # 4. grounding
        grounded = rails.verify_grounding(answer, chunks)
    """

    def __init__(self, min_retrieval_score: float | None = None) -> None:
        self.min_retrieval_score: float = min_retrieval_score or float(
            os.environ.get("MIN_RETRIEVAL_SCORE", "0.3")
        )

    # -----------------------------------------------------------------------
    # §11.1  Input Validation
    # -----------------------------------------------------------------------
    def validate_input(self, transcript: str) -> None:
        """Raise InputValidationError if transcript is empty, malformed, or too long.

        Called BEFORE expensive retrieval/generation stages.
        """
        if not transcript or not transcript.strip():
            raise InputValidationError(
                "Empty transcript. No speech was detected or the audio was too short."
            )
        if len(transcript) > MAX_QUERY_LENGTH:
            raise InputValidationError(
                f"Query is too long ({len(transcript)} chars). Maximum is {MAX_QUERY_LENGTH} characters."
            )
        logger.debug("Input validation passed: %r", transcript[:60])

    # -----------------------------------------------------------------------
    # §11.2  Off-Topic Detection
    # -----------------------------------------------------------------------
    def check_off_topic(self, query: str) -> None:
        """Raise OffTopicError if the query is clearly unrelated to the knowledge base.

        This is a heuristic check designed to refuse obviously unrelated questions
        BEFORE calling the expensive retrieval + generation pipeline.

        The knowledge base is MSMARCO-XI (general QA passages).
        Off-topic examples from PRD §8: weather, personal advice, code generation.

        This check is intentionally conservative — borderline queries proceed
        to retrieval where low confidence will catch them.
        """
        query_lower = query.lower().strip()

        off_topic_signals = [
            # Transient/real-time information unavailable in any static corpus
            r"\b(weather|forecast|temperature) (in|at|for|tomorrow|today)\b",
            # Personal / action requests outside QA
            r"\b(book (me|a|my|an)|order|buy|purchase|schedule|remind me)\b",
            # Code / programming tasks not in scope
            r"\b(write (me )?(a |an )?(\w+ )?(function|class|script|program|code))\b",
        ]

        for pattern in off_topic_signals:
            if re.search(pattern, query_lower):
                logger.info("Off-topic query detected: %r", query[:80])
                raise OffTopicError(
                    "I can only answer questions supported by the provided knowledge base."
                )

    # -----------------------------------------------------------------------
    # §11.4  Unsafe Input Detection
    # -----------------------------------------------------------------------
    def check_unsafe(self, text: str) -> None:
        """Raise UnsafeInputError if the text contains unsafe/jailbreak patterns.

        Called on the raw transcript before any other processing.
        """
        if _UNSAFE_PATTERNS.search(text):
            logger.warning("Unsafe input detected (pattern match).")
            raise UnsafeInputError("I can't assist with that request.")

    # -----------------------------------------------------------------------
    # §11.3  Low Retrieval Confidence
    # -----------------------------------------------------------------------
    def is_context_sufficient(self, chunks: List[RetrievedChunk]) -> bool:
        """Return True if at least one chunk meets the minimum relevance threshold.

        Returns False (rather than raising) so the orchestrator decides the response.
        Threshold is configurable via MIN_RETRIEVAL_SCORE env var (Tech Doc §11.3).
        """
        if not chunks:
            logger.info("Retrieval returned zero chunks — context insufficient.")
            return False

        best_score = max(c.score for c in chunks)
        sufficient = best_score >= self.min_retrieval_score
        if not sufficient:
            logger.info(
                "Best retrieval score %.3f is below threshold %.3f — context insufficient.",
                best_score,
                self.min_retrieval_score,
            )
        return sufficient

    # -----------------------------------------------------------------------
    # §11.5  Grounding Verification
    # -----------------------------------------------------------------------
    def verify_grounding(self, answer: str, chunks: List[RetrievedChunk]) -> bool:
        """Check whether the generated answer is grounded in the retrieved context.

        Strategy: lightweight lexical overlap check.
        If the LLM returned the INSUFFICIENT_CONTEXT sentinel, fail immediately.
        Otherwise check whether answer shares meaningful terms with context text.

        Returns True if grounded, False if the answer appears unsupported.
        """
        if not answer or answer.strip() == "INSUFFICIENT_CONTEXT":
            logger.info("Grounding check: LLM returned insufficient-context sentinel.")
            return False

        if not chunks:
            logger.info("Grounding check: no context chunks — cannot be grounded.")
            return False

        # Build a lowercase token set from all context text
        context_text = " ".join(c.text for c in chunks).lower()
        context_tokens = set(re.findall(r"\b\w{4,}\b", context_text))  # words ≥ 4 chars

        # Extract meaningful tokens from the answer
        answer_tokens = set(re.findall(r"\b\w{4,}\b", answer.lower()))

        if not answer_tokens:
            return False

        # Overlap ratio: what fraction of answer tokens appear in context?
        overlap = len(answer_tokens & context_tokens) / len(answer_tokens)
        grounded = overlap >= 0.25  # At least 25% of answer terms must appear in context

        logger.debug(
            "Grounding check: overlap=%.2f, grounded=%s", overlap, grounded
        )
        return grounded
