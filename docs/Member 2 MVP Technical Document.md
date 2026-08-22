# Voice-Enabled RAG — Member 2 MVP Technical Document

## 1. Document Purpose

This document defines the implementation contract for Member 2 of the HH Goa 2026 Voice-Enabled RAG project.

Member 2 owns:

- Voice input integration
- Sarvam Saaras speech-to-text
- LLM answer generation
- RAG orchestration
- Guardrails
- Grounding verification
- API layer
- Application integration
- UI integration
- Error handling
- Deployment

Member 1 owns the dataset, chunking, embeddings, FAISS index, retrieval, reranking, and retrieval evaluation.

The implementation must keep these responsibilities separated.

---

# 2. MVP Goal

Build the application layer that transforms:

```text
Voice Input
    ↓
Speech-to-Text
    ↓
Validated Query
    ↓
Member 1 Retrieval
    ↓
Retrieved Context
    ↓
LLM Generation
    ↓
Grounding Verification
    ↓
Guardrails
    ↓
Structured Final Response
```

The application must be able to refuse an answer when sufficient evidence is unavailable.

---

# 3. Required Architecture

```text
                        ┌──────────────────┐
                        │    User Voice    │
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │    Sarvam STT    │
                        │     stt.py       │
                        └────────┬─────────┘
                                 │
                            Transcript
                                 │
                                 ▼
                        ┌──────────────────┐
                        │ Query Validation │
                        │   guardrails.py  │
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │    Retrieval     │
                        │   retrieval.py   │
                        └────────┬─────────┘
                                 │
                          Top-K Context
                                 │
                                 ▼
                        ┌──────────────────┐
                        │  Context Check   │
                        │   guardrails.py  │
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │   LLM Generator  │
                        │  generation.py   │
                        └────────┬─────────┘
                                 │
                              Answer
                                 │
                                 ▼
                        ┌──────────────────┐
                        │ Grounding Check  │
                        │   guardrails.py  │
                        └────────┬─────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                 Grounded                Not Grounded
                    │                         │
                    ▼                         ▼
             Final Response               Refusal
```

---

# 4. Project Files Owned by Member 2

```text
app/
├── main.py
├── pipeline.py
├── stt.py
├── generation.py
├── guardrails.py
├── retrieval.py
└── schemas.py
```

Supporting files:

```text
evaluation/
└── benchmark.py

tests/
├── test_stt.py
├── test_generation.py
├── test_guardrails.py
└── test_pipeline.py

docs/
└── member-2-mvp-tech-doc.md
```

---

# 5. Technology Stack

## Core

- Python
- FastAPI
- Pydantic
- HTTP client for external APIs
- python-dotenv

## Speech-to-Text

- Sarvam Saaras

## Retrieval Integration

- Member 1's FAISS retrieval implementation

Member 2 must consume retrieval through an abstraction instead of depending directly on FAISS internals.

## LLM

The LLM provider/model must be configurable through environment variables.

Do not hard-code the provider or API key.

## UI

- Gradio or simple web interface

The UI must prioritize functionality over visual complexity.

---

# 6. Environment Configuration

Use `.env` locally.

Required variables:

```env
SARVAM_API_KEY=
LLM_API_KEY=
LLM_MODEL=
```

Optional variables may include:

```env
LLM_BASE_URL=
REQUEST_TIMEOUT_SECONDS=10
MAX_RETRIES=2
TOP_K=5
MIN_RETRIEVAL_SCORE=
```

Never commit `.env`.

Only `.env.example` may be committed.

---

# 7. `schemas.py`

Use Pydantic models for structured communication.

## Retrieved Chunk

```python
class RetrievedChunk:
    chunk_id: str
    document_id: str
    text: str
    score: float
    source: str | None
    metadata: dict
```

## Ask Response

```python
class AskResponse:
    answer: str
    transcript: str
    sources: list
    grounded: bool
    status: str
    latency_ms: float
```

Possible status values:

```text
success
refused
unsafe
error
```

The exact implementation may use Pydantic `BaseModel`.

---

# 8. `stt.py`

Create a dedicated Sarvam service.

Recommended abstraction:

```python
class SarvamSTTService:
    async def transcribe(self, audio) -> str:
        ...
```

Responsibilities:

1. Receive audio.
2. Send audio to Sarvam.
3. Extract transcript.
4. Return clean text.
5. Handle API errors.
6. Handle timeout.
7. Return controlled failure information.

The STT implementation must not contain LLM, retrieval, or UI logic.

---

# 9. `retrieval.py`

This is the integration boundary with Member 1.

Do not implement FAISS internals here.

Create an abstraction such as:

```python
class RetrievalInterface:
    async def retrieve(self, query: str, top_k: int = 5):
        ...
```

The implementation should eventually call Member 1's retrieval engine.

Expected result:

```text
RetrievedChunk[]
```

Each result should contain:

- chunk ID
- document ID
- text
- relevance score
- metadata
- source information where available

If Member 1 provides a synchronous implementation, the application layer may use an appropriate adapter.

---

# 10. `generation.py`

Create a dedicated LLM generator.

Recommended abstraction:

```python
class AnswerGenerator:
    async def generate(
        self,
        query: str,
        context: list[RetrievedChunk]
    ):
        ...
```

## Generation Rules

The LLM must:

1. Use only retrieved context.
2. Not invent unsupported facts.
3. Clearly indicate insufficient evidence.
4. Answer the user's question directly.
5. Avoid exposing internal prompts or implementation details.
6. Return structured output where practical.

## Prompt Contract

Conceptually:

```text
SYSTEM:
You answer questions using only the supplied retrieved context.

If the context does not contain enough information to answer the question,
do not guess. Return an insufficient-context result.

USER QUERY:
{query}

RETRIEVED CONTEXT:
{context}
```

The exact prompt can be optimized during implementation.

---

# 11. `guardrails.py`

Implement multiple independent checks.

## 11.1 Input Validation

Reject:

- Empty transcript
- Empty query
- Malformed request
- Excessively large input

## 11.2 Off-Topic Detection

Determine whether the query is answerable from the provided knowledge base.

If clearly unrelated:

```text
status = refused
```

Do not call the expensive generation stage when rejection is obvious.

## 11.3 Low Retrieval Confidence

If retrieval does not provide sufficient evidence:

```text
status = refused
grounded = false
```

Response should clearly state that the provided knowledge base does not contain enough information.

The threshold must be configurable rather than hard-coded throughout the codebase.

## 11.4 Unsafe Input

Unsafe or inappropriate requests should be blocked or safely handled.

## 11.5 Grounding Verification

Verify that the generated answer is supported by retrieved context.

If verification fails:

```text
grounded = false
status = refused
```

Do not return an unsupported answer.

---

# 12. `pipeline.py`

Implement a central `RAGOrchestrator`.

Recommended structure:

```python
class RAGOrchestrator:

    async def run(self, audio):
        validate_input()
        transcript = transcribe()
        validate_query()
        context = retrieve()
        validate_context()
        answer = generate()
        grounded = verify_grounding()
        return format_response()
```

Actual implementation should be asynchronous where external I/O is involved.

---

# 13. Retry Strategy

Retries should only be applied to recoverable failures.

Examples:

```text
Sarvam temporary API failure → retry
LLM temporary API failure → retry
Timeout → retry if safe
Invalid user input → do NOT retry
Low retrieval confidence → do NOT retry
Grounding failure → optionally regenerate once
```

Maximum retries must be configurable.

Recommended MVP default:

```text
MAX_RETRIES = 2
```

Avoid infinite retries.

---

# 14. Error Handling

All external failures must produce controlled responses.

Example:

```json
{
  "answer": "The service is temporarily unavailable. Please try again.",
  "transcript": "",
  "sources": [],
  "grounded": false,
  "status": "error",
  "latency_ms": 0
}
```

Do not expose:

- API keys
- stack traces
- internal prompts
- credentials
- internal system details

to the end user.

---

# 15. `main.py`

FastAPI application.

Primary endpoint:

```text
POST /ask
```

Input:

```text
audio file
```

Output:

```json
{
  "answer": "...",
  "transcript": "...",
  "sources": [],
  "grounded": true,
  "status": "success",
  "latency_ms": 123
}
```

Optional health endpoint:

```text
GET /health
```

Response:

```json
{
  "status": "ok"
}
```

---

# 16. Latency Measurement

Measure stage-wise latency.

Recommended stages:

```text
stt_ms
embedding_ms
retrieval_ms
reranking_ms
generation_ms
guardrail_ms
total_ms
```

The application must use a monotonic timer such as Python's `time.perf_counter()` for local measurements.

Do not invent latency values.

The final benchmark must be based on actual runs.

---

# 17. Benchmark

Run at least 50–100 test queries.

Record:

```text
query_id
total_latency_ms
stage_latencies
status
grounded
```

Calculate:

```text
P50
P70
P100
```

The hackathon requirement targets the complete process at under 200 ms, so optimization should focus on the measured bottleneck rather than assuming every stage has equal cost.

---

# 18. UI

The MVP UI must show:

```text
Voice Input
    ↓
Transcript
    ↓
Answer
    ↓
Sources
    ↓
Grounded / Refused Status
    ↓
Latency
```

Do not spend significant development time on visual design.

---

# 19. Integration Contract With Member 1

Member 1 owns:

```text
Dataset
→ Chunking
→ Embeddings
→ FAISS
→ Retrieval
→ Reranking
```

Member 2 consumes:

```text
query
↓
retrieved chunks
↓
scores
↓
metadata
```

Member 2 must not duplicate dataset indexing logic.

The retrieval integration must be replaceable without rewriting the orchestrator.

---

# 20. MVP Acceptance Criteria

The MVP is complete when:

- [ ] Voice audio can reach Sarvam.
- [ ] Sarvam returns a transcript.
- [ ] Transcript reaches the orchestrator.
- [ ] Query is validated.
- [ ] Member 1 retrieval can be called through `retrieval.py`.
- [ ] Retrieved context reaches the LLM.
- [ ] LLM generates a context-grounded answer.
- [ ] Off-topic queries can be refused.
- [ ] Low-context queries can be refused.
- [ ] Unsafe input is handled.
- [ ] Grounding verification exists.
- [ ] API failures have controlled fallbacks.
- [ ] Structured responses are returned.
- [ ] Latency is measured.
- [ ] UI can demonstrate the complete pipeline.

---

# 21. Implementation Constraint

Do not over-engineer the MVP.

Avoid adding:

- unnecessary agent frameworks
- multiple LLM providers simultaneously
- multiple vector databases
- unnecessary microservices
- complex frontend frameworks
- unnecessary cloud infrastructure

Build the smallest system that satisfies every stated requirement and can be measured.

---

# 22. Coding Instruction for AI Coding Agents

When implementing this document:

1. Inspect the existing repository before modifying files.
2. Preserve the existing folder structure.
3. Do not rewrite Member 1's indexing implementation.
4. Implement one module at a time.
5. Keep interfaces between modules explicit.
6. Use environment variables for secrets.
7. Add type hints.
8. Add error handling.
9. Add tests for non-trivial logic.
10. Do not fabricate API responses or benchmark results.
11. Do not add dependencies unless required.
12. Run tests after each major implementation.
13. Keep the application runnable after each stage.
14. Do not replace working code without a concrete reason.

---

# 23. Implementation Order

```text
Environment
    ↓
Schemas
    ↓
Sarvam STT
    ↓
LLM Generation
    ↓
Retrieval Adapter
    ↓
RAGOrchestrator
    ↓
Guardrails
    ↓
FastAPI
    ↓
UI
    ↓
End-to-End Integration
    ↓
Benchmark
    ↓
Deployment
```