# Member 2 Architecture Documentation

## Overview
Member 2 owns the Voice, AI, orchestration, guardrails, API, UI, deployment, and application integration layer for the Voice-Enabled RAG system.

## Components

### `app/main.py`
FastAPI entry point defining primary endpoints like `POST /ask`.

### `app/stt.py`
Integration with Sarvam Saaras speech-to-text service.

### `app/generation.py`
LLM answer generation given user query and retrieved context chunks.

### `app/guardrails.py`
Safety, topic validation, low-confidence refusal, and grounding check logic.

### `app/retrieval.py`
Integration boundary to consume Member 1's FAISS/Retrieval outputs.

### `app/pipeline.py`
Central `RAGOrchestrator` managing component flow, retry/error handling, and timing.

### `app/schemas.py`
Pydantic data contracts across API, orchestration, and component boundaries.

### `evaluation/benchmark.py`
Latency evaluation measuring P50, P70, and P100 metrics.
