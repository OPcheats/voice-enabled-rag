# Voice-Enabled RAG — Member 2 Product Requirements Document

## 1. Product

### Product Name

Voice-Enabled RAG

### Hackathon

HH Goa 2026

### Product Type

Voice-enabled Retrieval-Augmented Generation application.

---

# 2. Product Objective

Allow a user to ask a question using their voice and receive an answer grounded in the provided MSMARCO-XI knowledge base.

The system must know when it does not have sufficient information and refuse instead of hallucinating.

---

# 3. User Journey

```text
User opens application
        ↓
User records a question
        ↓
System transcribes voice
        ↓
System displays transcript
        ↓
System validates query
        ↓
System retrieves relevant knowledge
        ↓
System generates answer
        ↓
System verifies grounding
        ↓
System displays answer + sources + status + latency
```

---

# 4. Primary User Story

**As a user,**

I want to ask a question by voice,

so that I can receive a fast answer based only on the provided knowledge base.

---

# 5. Secondary User Stories

### Voice

As a user, I want my spoken question to be converted into text accurately.

### Grounded Answer

As a user, I want the answer to be based on retrieved information rather than unsupported generation.

### Sources

As a user, I want to see the context/source information used for the answer.

### Refusal

As a user, I want the system to clearly tell me when it does not have enough information.

### Safety

As a user, I want inappropriate or unsafe requests to be handled safely.

### Performance

As a user, I want the system to respond quickly.

---

# 6. Functional Requirements

## FR-01 Voice Input

The application must accept user voice input.

### Acceptance Criteria

- User can provide an audio recording.
- Audio reaches the STT service.
- Errors are handled gracefully.

---

## FR-02 Speech-to-Text

The system must use Sarvam Saaras for speech-to-text.

### Acceptance Criteria

- Audio is sent to Sarvam.
- Transcript is returned.
- Transcript is available to downstream stages.
- Sarvam failures produce controlled errors.

---

## FR-03 Query Validation

The system must validate the transcript before expensive processing.

### Acceptance Criteria

Reject:

- empty transcript
- malformed input
- clearly invalid input

---

## FR-04 Retrieval Integration

The system must send the validated query to Member 1's retrieval system.

### Acceptance Criteria

The retrieval layer returns:

- relevant chunks
- relevance scores
- metadata
- source information where available

---

## FR-05 Context Validation

The system must determine whether retrieved context is sufficient.

### Acceptance Criteria

If retrieval confidence is below the configured threshold:

```text
Do not generate an unsupported answer.
```

Return a refusal/insufficient-context response.

---

## FR-06 Answer Generation

The system must generate an answer using retrieved context.

### Acceptance Criteria

The answer:

- addresses the query
- uses retrieved context
- does not intentionally invent unsupported information
- handles insufficient context

---

## FR-07 Grounding Verification

The system must verify that the generated answer is supported by retrieved context.

### Acceptance Criteria

If the answer is unsupported:

```text
grounded = false
status = refused
```

The unsupported answer must not be presented as a valid grounded response.

---

## FR-08 Off-Topic Detection

The system must refuse questions unrelated to the available knowledge base.

Example:

```text
User:
"What is the weather in Goa tomorrow?"

System:
"I can't answer that from the provided knowledge base."
```

The exact wording may vary.

---

## FR-09 Unsafe Input

The system must detect and safely handle inappropriate or unsafe requests.

### Acceptance Criteria

Unsafe input must not cause the system to bypass its instructions or reveal internal information.

---

## FR-10 Error Recovery

The system must handle external API/model failures.

### Acceptance Criteria

- Retry recoverable failures.
- Do not retry invalid user input indefinitely.
- Return a useful fallback when recovery fails.
- Never expose secrets or stack traces.

---

## FR-11 Structured Response

The backend must return a predictable response.

Required conceptual fields:

```json
{
  "answer": "...",
  "transcript": "...",
  "sources": [],
  "grounded": true,
  "status": "success",
  "latency_ms": 0
}
```

---

## FR-12 Latency Monitoring

The system must record stage-wise latency and total latency.

Required measurements:

- STT
- embedding
- retrieval
- reranking
- generation
- guardrails
- total

---

## FR-13 Benchmarking

The system must be tested against a reasonable batch of queries.

Target:

```text
50–100+ queries
```

Required metrics:

```text
P50
P70
P100
```

Actual measured values must be used.

---

## FR-14 UI

The application should expose:

- voice recording
- transcript
- answer
- sources/context
- grounded status
- latency

The UI should be simple and demonstration-focused.

---

# 7. Non-Functional Requirements

## NFR-01 Latency

The hackathon specifies an under-200-ms target for the full process.

The team must measure actual latency and identify bottlenecks.

Do not claim compliance without benchmark evidence.

---

## NFR-02 Reliability

Temporary external API/model failures should be handled with controlled retries and fallback responses.

---

## NFR-03 Security

Secrets must:

- remain in environment variables
- never be committed to Git
- never be returned in API responses

---

## NFR-04 Maintainability

The implementation must use separate modules for:

```text
STT
Retrieval
Generation
Guardrails
Orchestration
API
Schemas
```

---

## NFR-05 Replaceability

LLM and retrieval implementations should be replaceable without rewriting the entire application.

---

# 8. User-Facing States

## Success

```text
Status: Grounded
Answer: ...
Sources: ...
Latency: ... ms
```

## Insufficient Context

```text
Status: Not enough information

I couldn't find enough relevant information in
the provided knowledge base to answer this question.
```

## Off Topic

```text
Status: Refused

I can only answer questions supported by the
provided knowledge base.
```

## Unsafe

```text
Status: Blocked

I can't assist with that request.
```

## Service Error

```text
Status: Error

The service is temporarily unavailable.
Please try again.
```

---

# 9. API Requirements

## `GET /health`

Purpose:

Check whether the application is running.

Response:

```json
{
  "status": "ok"
}
```

---

## `POST /ask`

Purpose:

Process a voice question through the complete pipeline.

Conceptual flow:

```text
audio
 ↓
Sarvam STT
 ↓
validation
 ↓
retrieval
 ↓
context validation
 ↓
LLM
 ↓
grounding
 ↓
response
```

Response:

```json
{
  "answer": "...",
  "transcript": "...",
  "sources": [],
  "grounded": true,
  "status": "success",
  "latency_ms": 0
}
```

---

# 10. UI Requirements

The UI should contain:

```text
Voice Recording
      ↓
Transcript
      ↓
Answer
      ↓
Sources
      ↓
Grounded Status
      ↓
Latency
```

A technical/debug view may additionally expose stage-level latency.

Do not prioritize visual polish over pipeline functionality.

---

# 11. Grounding Product Behavior

The most important product rule is:

> The system must prefer refusing to answer over generating an unsupported answer.

Decision:

```text
Retrieved Context
       │
       ▼
Is context sufficient?
   │          │
  NO         YES
   │          │
   ▼          ▼
 REFUSE      Generate
               │
               ▼
          Grounding Check
             │       │
            NO      YES
             │       │
             ▼       ▼
           REFUSE   ANSWER
```

---

# 12. Demo Requirements

The final demonstration should show:

1. Voice recording.
2. Speech transcription.
3. Retrieved context/sources.
4. Generated answer.
5. Grounded status.
6. Latency.
7. An unanswerable/off-topic query.
8. The system refusing the unanswerable query.

---

# 13. MVP Scope

## In Scope

- Sarvam STT
- RAG retrieval integration
- LLM generation
- RAG orchestration
- Guardrails
- Grounding check
- FastAPI API
- Simple UI
- Error handling
- Latency measurement
- Benchmarking
- Deployment

## Out of Scope

- Complex multi-agent architecture
- Multiple vector databases
- Complex frontend animations
- User accounts
- Persistent chat history
- Analytics dashboard
- Production billing
- Multi-tenant infrastructure
- Unnecessary microservices

---

# 14. Definition of Done

The Member 2 implementation is complete when:

- [ ] User can submit voice.
- [ ] Sarvam returns a transcript.
- [ ] Transcript is validated.
- [ ] Retrieval integration works.
- [ ] Retrieved context reaches the LLM.
- [ ] LLM generates a context-based answer.
- [ ] Grounding is verified.
- [ ] Off-topic queries are refused.
- [ ] Low-context queries are refused.
- [ ] Unsafe inputs are handled.
- [ ] External failures have fallback behavior.
- [ ] Structured API response works.
- [ ] UI displays the complete result.
- [ ] Latency is measured.
- [ ] Benchmark can produce P50/P70/P100.
- [ ] End-to-end demo works.
- [ ] Deployment works.

---

# 15. AI Coding Agent Instructions

Any coding agent working on Member 2 should follow these rules:

1. Read this PRD before implementation.
2. Read `member-2-mvp-tech-doc.md` before implementation.
3. Inspect the existing repository first.
4. Preserve the existing folder structure.
5. Do not modify Member 1's indexing/chunking implementation.
6. Use the retrieval interface defined by `retrieval.py`.
7. Keep secrets in `.env`.
8. Do not hard-code API keys.
9. Do not fabricate API responses.
10. Do not fabricate latency benchmarks.
11. Keep dependencies minimal.
12. Use async I/O for external API calls where appropriate.
13. Add error handling around external services.
14. Add tests for important logic.
15. Keep the application runnable after each implementation stage.
16. Do not introduce unnecessary frameworks.
17. Do not rewrite existing working code without justification.
18. If a required external API detail is unknown, inspect the existing implementation/configuration or official documentation instead of guessing.
19. Prefer simple, testable modules over large files.
20. Before finishing a task, run the relevant tests and report failures honestly.

---

# 16. Implementation Sequence

```text
1. Project Environment
        ↓
2. Schemas
        ↓
3. Sarvam STT
        ↓
4. LLM Generation
        ↓
5. Retrieval Adapter
        ↓
6. RAGOrchestrator
        ↓
7. Guardrails
        ↓
8. FastAPI
        ↓
9. UI
        ↓
10. Integration
        ↓
11. Testing
        ↓
12. Benchmark
        ↓
13. Deployment
```