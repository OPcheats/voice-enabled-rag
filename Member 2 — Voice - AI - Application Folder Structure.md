# Member 2 — Voice / AI / Application

## 1. Responsibility

Member 2 owns the Voice, AI, orchestration, guardrails, API, UI, deployment, and application integration layer.

The responsibility starts after user voice input and connects the retrieval system built by Member 1 with the final user-facing response.

## 2. Folder Structure

```text
app/
│
├── __init__.py
│
├── main.py
│   └── FastAPI application entry point
│
├── pipeline.py
│   └── Central RAGOrchestrator
│
├── stt.py
│   └── Sarvam Saaras speech-to-text integration
│
├── generation.py
│   └── LLM answer generation using retrieved context
│
├── guardrails.py
│   └── Query safety, relevance, context validation,
│       grounding and refusal logic
│
├── schemas.py
│   └── Pydantic request/response models
│
└── retrieval.py
    └── Interface for connecting Member 1's retrieval system
```

## 3. Supporting Folders

```text
evaluation/
└── benchmark.py
    └── End-to-end latency benchmarking

tests/
├── test_stt.py
├── test_generation.py
├── test_guardrails.py
└── test_pipeline.py

docs/
└── member-2-architecture.md
```

## 4. Ownership

### `main.py`

Responsible for the application/API entry point.

Responsibilities:

- Start FastAPI
- Define API endpoints
- Receive user requests
- Return structured responses

Primary endpoint:

```text
POST /ask
```

---

### `stt.py`

Responsible for Sarvam Saaras integration.

Flow:

```text
Voice Audio
    ↓
Sarvam Saaras
    ↓
Transcript
```

Responsibilities:

- Send audio to Sarvam
- Receive transcription
- Handle API errors
- Handle timeout/failure cases

---

### `generation.py`

Responsible for LLM-based answer generation.

Input:

```text
User Query
+
Retrieved Context
```

Output:

```text
Generated Answer
```

The LLM must be instructed to answer using the retrieved context rather than inventing unsupported information.

---

### `pipeline.py`

This is the central `RAGOrchestrator`.

Pipeline:

```text
Validate Input
      ↓
Transcribe Audio
      ↓
Validate / Classify Query
      ↓
Retrieve Context
      ↓
Validate Context
      ↓
Generate Answer
      ↓
Verify Grounding
      ↓
Format Response
```

The orchestrator is responsible for:

- Calling each pipeline stage
- Passing structured data between stages
- Retry handling
- Timeout handling
- Error recovery
- Returning a consistent final response

---

### `guardrails.py`

Responsible for controlling when the system should answer and when it should refuse.

Required cases:

```text
Off-topic Query
      ↓
    REFUSE

Low Retrieval Confidence
      ↓
    REFUSE

Insufficient Context
      ↓
    REFUSE

Unsafe / Inappropriate Input
      ↓
    BLOCK / SAFE RESPONSE

Unsupported Generated Answer
      ↓
    REJECT / REVISE
```

The system should prefer refusing over hallucinating when sufficient evidence is unavailable.

---

### `schemas.py`

Contains structured Pydantic models.

Example response:

```json
{
  "answer": "Generated answer",
  "sources": [],
  "grounded": true,
  "status": "success",
  "latency_ms": 120
}
```

Schemas should keep communication between the STT, retrieval, generation, guardrail, API, and UI layers consistent.

---

### `retrieval.py`

This is the integration boundary with Member 1.

Member 1 owns:

```text
Dataset
   ↓
Chunking
   ↓
Embeddings
   ↓
FAISS
   ↓
Retrieval
```

Member 2 consumes the retrieval result through a clean interface.

Expected interface:

```python
retrieve(query)
```

Expected output:

```text
Retrieved Chunks
+
Scores
+
Metadata
+
Sources
```

Member 2 should not modify the internal FAISS/indexing implementation unless required during integration.

## 5. Application Data Flow

```text
                USER
                  │
                  ▼
             Voice Input
                  │
                  ▼
              stt.py
                  │
                  ▼
             Transcript
                  │
                  ▼
           pipeline.py
        RAGOrchestrator
                  │
                  ▼
            Guardrails
                  │
                  ▼
          retrieval.py
                  │
                  ▼
       Member 1 Retrieval
                  │
                  ▼
        Retrieved Context
                  │
                  ▼
          generation.py
                  │
                  ▼
            LLM Answer
                  │
                  ▼
          Grounding Check
                  │
           ┌──────┴──────┐
           │             │
        Grounded      Not Grounded
           │             │
           ▼             ▼
        Answer         Refuse
           │
           ▼
          main.py
           │
           ▼
       Final Response
```

## 6. Error Handling

The application should handle:

- Sarvam API failure
- LLM API failure
- Retrieval failure
- Timeout
- Empty transcript
- Invalid query
- Off-topic query
- Insufficient retrieved context
- Grounding failure

External API/model failures should use retries where appropriate and return a clear fallback response when recovery is not possible.

## 7. Latency Monitoring

Member 2 tracks application-level latency.

Stages:

```text
STT
Embedding
Retrieval
Reranking
Generation
Guardrails
Total End-to-End
```

The final benchmark must calculate:

```text
P50
P70
P100
```

using a reasonable batch of test queries.

Actual measured values must be used in the final submission.

## 8. UI Integration

The UI should expose:

```text
┌─────────────────────────────┐
│       Voice Input           │
│          🎙️                 │
├─────────────────────────────┤
│ Transcript                  │
├─────────────────────────────┤
│ Answer                      │
├─────────────────────────────┤
│ Sources                     │
├─────────────────────────────┤
│ Grounded Status             │
├─────────────────────────────┤
│ Pipeline Latency            │
└─────────────────────────────┘
```

The UI should remain simple and focus on demonstrating the complete pipeline rather than visual complexity.

## 9. Member 2 Development Order

```text
1. Environment Setup
        ↓
2. Sarvam STT
        ↓
3. LLM Generation
        ↓
4. Pydantic Schemas
        ↓
5. RAGOrchestrator
        ↓
6. Guardrails
        ↓
7. Retrieval Integration
        ↓
8. FastAPI
        ↓
9. UI
        ↓
10. End-to-End Testing
        ↓
11. Latency Benchmark
        ↓
12. Deployment
```

## 10. Boundary With Member 1

### Member 1 provides:

```text
Query
   ↓
Retrieval Engine
   ↓
Top-K Relevant Chunks
   +
Scores
   +
Metadata
   +
Sources
```

### Member 2 consumes:

```text
Query
+
Retrieved Context
```

and produces:

```text
Grounded Answer
+
Sources
+
Status
+
Latency
```

This separation allows both members to develop independently and integrate the components later.