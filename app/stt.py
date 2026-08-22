"""
Sarvam Saaras Speech-to-Text integration — Tech Doc §8.

Responsibilities:
  1. Receive audio bytes and filename/MIME hint.
  2. Send audio to Sarvam via multipart/form-data HTTP POST using model="saaras:v3".
  3. Extract transcript from response.
  4. Return clean text string.
  5. Handle API errors with typed exceptions and safe diagnostic logging.
  6. Handle timeouts and retries.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sarvam STT REST API endpoint (Official Sarvam AI API)
# https://api.sarvam.ai/speech-to-text
# ---------------------------------------------------------------------------
SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"


class STTError(Exception):
    """Raised when Sarvam STT fails after all retries."""


class SarvamSTTService:
    """Async wrapper around official Sarvam Saaras STT REST API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        self.api_key: str = api_key if api_key is not None else os.environ.get("SARVAM_API_KEY", "")
        self.timeout: float = timeout or float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "10"))
        self.max_retries: int = max_retries or int(os.environ.get("MAX_RETRIES", "2"))

        # Step 2: Safe environment logging without exposing secrets
        is_configured = bool(self.api_key and self.api_key.strip())
        logger.info("SARVAM_API_KEY configured: %s", is_configured)

    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        content_type: Optional[str] = None,
    ) -> str:
        """Send audio bytes to Sarvam and return the transcript string.

        Args:
            audio_bytes: Raw audio file bytes.
            filename:    Filename hint for the multipart upload.
            content_type: Optional explicit MIME type (e.g. "audio/webm").

        Returns:
            Transcript string.

        Raises:
            STTError: If Sarvam is unavailable or returns an error response.
            ValueError: If SARVAM_API_KEY is not configured or audio_bytes is empty.
        """
        # Step 2: Validate API key configuration
        if not self.api_key or not self.api_key.strip():
            logger.error("STT failed: SARVAM_API_KEY is not configured.")
            raise ValueError(
                "SARVAM_API_KEY is not set. Add it to your .env file."
            )

        # Step 3: Validate audio payload size
        if not audio_bytes:
            logger.error("STT failed: Received empty audio payload (0 bytes).")
            raise ValueError("Audio payload is empty (0 bytes).")

        # Step 5: Safe diagnostic logging
        resolved_mime = content_type or self._detect_mime(filename)
        logger.info(
            "STT request started: filename=%s, content_type=%s, size_bytes=%d",
            filename,
            resolved_mime,
            len(audio_bytes),
        )

        last_error: Exception = RuntimeError("No attempts made")

        for attempt in range(1, self.max_retries + 2):
            try:
                transcript = await self._call_sarvam(audio_bytes, filename, resolved_mime)
                if attempt > 1:
                    logger.info("STT succeeded on attempt %d", attempt)
                logger.info(
                    "STT completed: parsed transcript length=%d chars", len(transcript)
                )
                return transcript

            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = exc
                logger.warning(
                    "STT transient network failure (attempt %d/%d): %s",
                    attempt,
                    self.max_retries + 1,
                    type(exc).__name__,
                )

            except httpx.HTTPStatusError as exc:
                err_detail = exc.response.text
                logger.error(
                    "Sarvam STT HTTP %d error: %s",
                    exc.response.status_code,
                    err_detail[:200] if err_detail else "No response body",
                )
                # 4xx errors (bad request, auth) are not retryable
                if exc.response.status_code < 500:
                    raise STTError(
                        f"Sarvam returned HTTP {exc.response.status_code}: {err_detail}"
                    ) from exc
                last_error = exc
                logger.warning(
                    "STT server error HTTP %d (attempt %d/%d)",
                    exc.response.status_code,
                    attempt,
                    self.max_retries + 1,
                )

        raise STTError(
            f"Sarvam STT failed after {self.max_retries + 1} attempts. Last error: {type(last_error).__name__}"
        ) from last_error

    async def _call_sarvam(
        self,
        audio_bytes: bytes,
        filename: str,
        content_type: str,
    ) -> str:
        """Execute a single HTTP call to Sarvam STT REST API."""
        logger.info("Sarvam request started to %s", SARVAM_STT_URL)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                SARVAM_STT_URL,
                headers={"api-subscription-key": self.api_key},
                files={"file": (filename, audio_bytes, content_type)},
                data={
                    "model": "saaras:v3",
                    "mode": "transcribe",
                },
            )

            logger.info(
                "Sarvam HTTP status: %d, response content_type: %s",
                response.status_code,
                response.headers.get("content-type", "unknown"),
            )

            response.raise_for_status()

            try:
                payload = response.json()
            except Exception as json_err:
                logger.error("Failed to parse JSON from Sarvam response: %s", json_err)
                raise STTError("Sarvam returned malformed non-JSON response.") from json_err

        # Step 6: Validate response structure per official Sarvam API spec
        if not isinstance(payload, dict):
            logger.error("Sarvam response payload is not a JSON object: %r", payload)
            raise STTError("Sarvam returned invalid JSON response format.")

        if "transcript" not in payload:
            logger.error("Sarvam response missing 'transcript' key: %r", payload)
            raise STTError("Sarvam response missing transcript field.")

        transcript: str = str(payload.get("transcript") or "").strip()
        logger.debug("Sarvam transcript extracted: %r", transcript[:80])
        return transcript

    @staticmethod
    def _detect_mime(filename: str) -> str:
        """Helper to resolve audio MIME type from file extension."""
        ext = filename.lower().split(".")[-1] if "." in filename else ""
        mime_map = {
            "wav": "audio/wav",
            "webm": "audio/webm",
            "mp3": "audio/mp3",
            "ogg": "audio/ogg",
            "flac": "audio/flac",
            "m4a": "audio/m4a",
        }
        return mime_map.get(ext, "audio/wav")
