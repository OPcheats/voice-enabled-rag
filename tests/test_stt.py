"""
Tests for SarvamSTTService — Tech Doc §8.

All tests use mocked HTTP responses. No real Sarvam API calls are made here.
Real API test is done independently via scratch/test_sarvam.py.
"""

from __future__ import annotations

import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch

from app.stt import SarvamSTTService, STTError


@pytest.fixture
def stt():
    return SarvamSTTService(api_key="test-key", timeout=5.0, max_retries=1)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_transcribe_success(stt):
    """Returns transcript string on successful Sarvam response."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = {"transcript": "What is machine learning?"}
    mock_response.raise_for_status = MagicMock()

    with patch("app.stt.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await stt.transcribe(b"fake-audio-bytes")

    assert result == "What is machine learning?"


@pytest.mark.asyncio
async def test_transcribe_empty_transcript_returns_empty_string(stt):
    """Empty transcript from Sarvam should return empty string."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = {"transcript": ""}
    mock_response.raise_for_status = MagicMock()

    with patch("app.stt.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await stt.transcribe(b"silent-audio")

    assert result == ""


# ---------------------------------------------------------------------------
# Validation & Error handling — Step 9
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_transcribe_empty_bytes_raises_value_error(stt):
    """Empty audio payload (0 bytes) raises ValueError before HTTP call."""
    with pytest.raises(ValueError, match="empty"):
        await stt.transcribe(b"")


@pytest.mark.asyncio
async def test_transcribe_timeout_retries_then_raises(stt):
    """Timeout is retryable. After max retries exhausted, raises STTError."""
    with patch("app.stt.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        with pytest.raises(STTError, match="Sarvam STT failed after"):
            await stt.transcribe(b"audio")


@pytest.mark.asyncio
async def test_transcribe_4xx_not_retried(stt):
    """4xx responses (e.g. 401 bad API key) should raise immediately without retry."""
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = "Unauthorized"
    http_error = httpx.HTTPStatusError("Unauthorized", request=MagicMock(), response=mock_response)

    with patch("app.stt.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=http_error)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        with pytest.raises(STTError, match="HTTP 401"):
            await stt.transcribe(b"audio")

    assert mock_client.post.call_count == 1


@pytest.mark.asyncio
async def test_transcribe_raises_if_no_api_key():
    """Should raise ValueError immediately if SARVAM_API_KEY is not configured."""
    stt = SarvamSTTService(api_key="", timeout=5.0, max_retries=0)
    with pytest.raises(ValueError, match="SARVAM_API_KEY"):
        await stt.transcribe(b"audio")
