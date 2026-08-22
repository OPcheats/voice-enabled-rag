"""
Tests for AnswerGenerator — Tech Doc §10.

All tests mock the OpenAI client. No real LLM calls are made.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.generation import AnswerGenerator, GenerationError, SYSTEM_PROMPT
from app.schemas import RetrievedChunk


@pytest.fixture
def mock_chunks():
    return [
        RetrievedChunk(
            chunk_id="c1",
            document_id="d1",
            text="Machine learning is a subset of artificial intelligence.",
            score=0.85,
            source="passage-1",
            metadata={},
        )
    ]


@pytest.fixture
def generator():
    with patch("app.generation.AsyncOpenAI"):
        gen = AnswerGenerator(api_key="test-key", model="gpt-4o-mini")
    return gen


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_generate_returns_answer(generator, mock_chunks):
    """Returns the LLM completion content as a string."""
    mock_choice = MagicMock()
    mock_choice.message.content = "Machine learning is a branch of AI."
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    generator._client.chat.completions.create = AsyncMock(return_value=mock_completion)

    result = await generator.generate("What is machine learning?", mock_chunks)
    assert result == "Machine learning is a branch of AI."


# ---------------------------------------------------------------------------
# Prompt construction — Tech Doc §10
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_generate_includes_query_and_context_in_user_message(generator, mock_chunks):
    """The user message must contain both the query and the context text."""
    captured_messages = []

    async def capture_create(**kwargs):
        captured_messages.extend(kwargs["messages"])
        mock_choice = MagicMock()
        mock_choice.message.content = "Some answer."
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        return mock_completion

    generator._client.chat.completions.create = capture_create

    await generator.generate("What is AI?", mock_chunks)

    system_msg = captured_messages[0]
    user_msg = captured_messages[1]

    assert system_msg["role"] == "system"
    assert "only" in system_msg["content"].lower()  # System prompt instructs use of context only
    assert "What is AI?" in user_msg["content"]
    assert "Machine learning is a subset" in user_msg["content"]


@pytest.mark.asyncio
async def test_generate_with_no_context_includes_no_context_placeholder(generator):
    """When no chunks provided, user message should note no context retrieved."""
    captured_messages = []

    async def capture_create(**kwargs):
        captured_messages.extend(kwargs["messages"])
        mock_choice = MagicMock()
        mock_choice.message.content = "INSUFFICIENT_CONTEXT"
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        return mock_completion

    generator._client.chat.completions.create = capture_create

    result = await generator.generate("What is AI?", [])
    assert "No context retrieved" in captured_messages[1]["content"]
    assert result == "INSUFFICIENT_CONTEXT"


# ---------------------------------------------------------------------------
# Error handling — Tech Doc §13
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_generate_raises_generation_error_after_retries(generator, mock_chunks):
    """Raises GenerationError after max retries for connection errors."""
    from openai import APIConnectionError
    generator.max_retries = 1

    generator._client.chat.completions.create = AsyncMock(
        side_effect=APIConnectionError.__new__(APIConnectionError)
    )

    with pytest.raises(GenerationError, match="failed after"):
        await generator.generate("query", mock_chunks)


def test_generator_raises_if_no_api_key():
    """Raises ValueError immediately if LLM_API_KEY is not set."""
    with patch.dict("os.environ", {"LLM_API_KEY": ""}, clear=False):
        with pytest.raises(ValueError, match="LLM_API_KEY"):
            AnswerGenerator(api_key="")
