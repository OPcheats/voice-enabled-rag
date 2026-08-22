# voice-enabled-rag
Voice-enabled Retrieval-Augmented Generation system using Sarvam STT, MSMARCO-XI, FAISS, and grounded LLM generation.
# Voice-Enabled RAG

A voice-enabled Retrieval-Augmented Generation system built for HH Goa 2026.

## Pipeline

Voice Input → Sarvam STT → Query Validation → Retrieval → LLM → Grounding Check → Answer

## Dataset

AI4Bharat MSMARCO-XI

## Tech Stack

- Python
- FastAPI
- Sarvam Saaras
- FAISS
- LLM
- Gradio
- Pydantic

## Project Structure

```text
app/          # Application and RAG pipeline
indexing/     # Dataset processing and indexing
data/         # Raw, processed and index data
evaluation/   # Benchmarking
tests/        # Automated tests
docs/         # Documentation
Team
Member 1 — RAG / Backend
Member 2 — Voice / AI / Application
