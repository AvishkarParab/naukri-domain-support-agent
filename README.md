# Naukri Domain Support Agent (Capstone)

A production-grade, zero-cost, deterministic enterprise customer and HR support system built for the Naukri recruitment ecosystem. The system features a multi-agent orchestration architecture adhering strictly to the **Principle of Least Autonomy**, grounded Retrieval-Augmented Generation (RAG) over structured HR policies, deterministic application lookups, algorithmic candidate escalation modeling, strict PII anonymization, and prompt injection defense.

---

## 1. Executive Summary & Design Highlights

- **Zero-Cost, Fully Local & Air-Gapped Execution:** Operates deterministically without reliance on third-party cloud LLM APIs, external tokens, or network calls. Fully telemetry-isolated (`CREWAI_DISABLE_TELEMETRY=true`, `OTEL_SDK_DISABLED=true`).
- **Least-Autonomy Architecture:** Workers (`Retrieval Agent`, `Lookup Agent`) hold isolated tools without open-ended generation autonomy. The `Response Composer` possesses strictly zero tools, mitigating unauthorized tool reuse and cyclic execution loops.
- **Continuous Algorithmic Escalation:** Eliminates ad-hoc heuristic routing by calculating continuous multi-variable escalation indices combining priority flags and normalized pending days.
- **Comprehensive Defense-in-Depth Guardrails:** Deterministic regex-based PII scrubbing (emails, phone numbers, application IDs) prior to inference, combined with proactive prompt injection interception and strict chunk-source ground truth verification.
- **In-Memory Grounded Response Caching & Runtime Budgeting:** Normalization-driven caching providing orders-of-magnitude latency speedups on frequent queries, paired with upfront token-budget bounding.
- **100% Evaluation Suite Attainment:** 15/15 unit and integration tests passing in `evaluate.py`, with 100% pass on scaling, cache speedup, and concurrent load tests in `scaling_eval.py`.

---

## 2. System Architecture

### High-Level End-to-End Request Pipeline

```
                       [Incoming User Query]
                                 │
                                 ▼
                     [Runtime Budget Check] ────(Exceeds 350 tokens)───► [BudgetExceededException]
                                 │
                                 ▼
                         [PII Masking] (Regex-based sanitization)
                                 │
                                 ▼
                   [Prompt Injection Detection] ───(Attack detected)───► [Refusal Schema (Escalated)]
                                 │
                                 ▼
                    [Session Memory Resolution] (LangChain InMemoryHistory)
                                 │
                                 ▼
                   [Query Routing & Context Set]
                                 │
             ┌───────────────────┴───────────────────┐
             ▼                                       ▼
  [Application ID Present]              [General / Policy Query]
             │                                       │
             ▼                                       ▼
    [Lookup Worker Agent]                  [Retrieval Worker Agent]
   (check_job_application_status)             (grounded_rag_query)
             │                                       │
             └───────────────────┬───────────────────┘
                                 │
                                 ▼
                      [Response Composer]
                 (Zero tools - Pure Synthesis)
                                 │
                                 ▼
                   [Groundedness Verification]
                                 │
                                 ▼
                 [Cache Layer (Set & Normalized)]
                                 │
                                 ▼
                    [AgentResponseSchema Output]
```

---

## 3. Mathematical Formulations & Bounding Logic

### A. Continuous Candidate Escalation Index

When a candidate application status is checked, the system calculates an automated priority score $\mathcal{S}_{\text{esc}} \in [0.0, 1.0]$ rather than using binary branching:

$$\mathcal{S}_{\text{esc}} = w_p \cdot \mathbb{I}(\text{priority}) + w_d \cdot \left(\frac{\min(d_{\text{pending}}, D_{\max})}{D_{\max}}\right)$$

Where:

- $w_p = 0.50$ (Priority flag weight)
- $w_d = 0.50$ (Pending duration weight)
- $\mathbb{I}(\text{priority}) \in \{0.0, 1.0\}$: Binary indicator whether `priority_review` is enabled
- $d_{\text{pending}}$: Integer count of elapsed days since application submission
- $D_{\max} = 30$: Maximum saturation threshold for pending days

**Escalation Decision Boundary:**
$$\text{recommend\_escalation} = \begin{cases} \mathbf{True}, & \text{if } \mathcal{S}_{\text{esc}} \ge 0.70 \\ \mathbf{False}, & \text{if } \mathcal{S}_{\text{esc}} < 0.70 \end{cases}$$

### B. Heuristic Token Estimation & Budget Bounding

To guard against context overflows and resource exhaustion before dispatching to agents, queries undergo deterministic pre-flight token budgeting:

$$\mathcal{T}_{\text{estimated}}(q) = \lfloor |W_q| \times 1.3 \rfloor$$

Where:

- $|W_q|$ is the whitespace-delimited word count of query $q$.
- Budget Invariant: $\mathcal{T}_{\text{estimated}}(q) \le \text{MAX\_TOKEN\_BUDGET\_PER\_REQUEST} = 350$.
- Violation triggers immediate dispatch abort via `BudgetExceededException`.

### C. Cache Latency Reduction Metric

Performance gains delivered by the normalization layer are measured across cold/warm evaluation cycles:

$$\text{Speedup Factor} = \frac{\overline{L}_{\text{cold}}}{\max(\overline{L}_{\text{warm}}, 10^{-4})}$$

---

## 4. Module Breakdown & File Organization

```
naukri-domain-support-agent/
│
├── .env                         # Environment constants and telemetry suppression
├── mock_llm.py                  # Deterministic LLM engine with context tracking
├── crew.py                      # Multi-agent definitions, tools, and orchestration
├── cache.py                     # Singleton grounded cache & token budget governor
├── rag_engine.py                # Local ChromaDB vector retrieval & policy matching
├── guardrails.py               # PII sanitization, prompt injection check, response schema
├── tools.py                     # Application DB connector and escalation arithmetic
├── evaluate.py                  # 15-test functional verification test harness
└── scaling_eval.py              # Concurrency, latency, and budget stress benchmarks
```

### Module Responsibilities

| Module            | Core Purpose                                                                          | Key Functions / Classes                                                                   |
| ----------------- | ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `mock_llm.py`     | Supplies predictable, zero-latency inference for local testing without network calls. | `DeterministicMockLLM`, `set_active_query_context()`, `extract_core_query()`              |
| `crew.py`         | Manages sequential CrewAI workflows adhering to the Least Autonomy pattern.           | `build_naukri_crew()`, `run_orchestrated_query()`, `InMemoryChatHistory`                  |
| `cache.py`        | Governs memory-efficient exact/normalized query reuse and budget caps.                | `GroundedResponseCache`, `enforce_runtime_budget()`, `estimate_token_count()`             |
| `tools.py`        | Encapsulates candidate database queries and escalation mathematics.                   | `check_job_application_status()`, escalation index calculator                             |
| `guardrails.py`   | Enforces structural input/output safety, sanitization, and output schemas.            | `mask_pii()`, `detect_prompt_injection()`, `verify_groundedness()`, `AgentResponseSchema` |
| `scaling_eval.py` | Validates concurrency limits, cache acceleration, and memory invariants.              | `benchmark_cache_speedup()`, `benchmark_burst_concurrency()`, `run_stress_suite()`        |

---

## 5. Agent Architecture: Principle of Least Autonomy

The system splits responsibilities across three discrete agents:

```
[User Query]
     │
     ├───► [Retrieval Agent]  Tools: [rag_lookup_tool]
     │                        Scope: Knowledge base clause matching only
     │
     ├───► [Lookup Agent]     Tools: [application_lookup_tool]
     │                        Scope: Record lookup & escalation calculation only
     │
     └───► [Response Composer] Tools: NONE (0 tools)
                              Scope: Format and synthesize final customer response
```

1. **Retrieval Agent:**
   - **Scope:** Restricted exclusively to HR policy document inspection.
   - **Allowed Tools:** `rag_lookup_tool` only.
2. **Lookup Agent:**
   - **Scope:** Restricted exclusively to recruitment record data extraction.
   - **Allowed Tools:** `application_lookup_tool` only.
3. **Response Composer:**
   - **Scope:** Pure synthesis. Formulates coherent, empathetic, grounded text from context generated by previous tasks.
   - **Allowed Tools:** **Zero tools assigned.** Prevents the composer from triggering unexpected tool loops or altering operational state during text synthesis.

---

## 6. Input/Output Specifications

### Request Ingestion Schema

Queries are accepted via `run_orchestrated_query(query: str, session_id: str = "default_session")`.

### Unified Response Schema (`AgentResponseSchema`)

```python
class AgentResponseSchema(BaseModel):
    query: str                # Sanitized query string with PII scrubbed
    status: str               # "success" | "refused"
    answer: str               # Synthesized response or security rejection notice
    sources: List[str]        # ["sentence_chunk_kb"] | ["JOB_APPLICATIONS_DB"]
    escalation_flag: bool     # True if escalation math threshold >= 0.70 or security refusal
```

---

## 7. Performance Benchmarks & Test Results

### Functional Test Suite (`evaluate.py`)

All 15 evaluation scenarios pass deterministically:

| Test Case | Scenario / Domain                   | Evaluated Criteria                       |  Result  |
| --------- | ----------------------------------- | ---------------------------------------- | :------: |
| TC-01     | Standard Notice Period              | Policy RAG Chunk Match                   | **PASS** |
| TC-02     | Casual Leave Entitlements           | Numeric Policy Clause Lookup             | **PASS** |
| TC-03     | Probation Extension Rules           | Grounded Policy Clause Lookup            | **PASS** |
| TC-04     | Valid Application ID (`APP-1001`)   | Database Match & Status Return           | **PASS** |
| TC-05     | Missing Application ID (`APP-9999`) | Graceful Missing Record Handling         | **PASS** |
| TC-06     | High Escalation Target              | Math Boundary $\ge 0.70 \to \text{True}$ | **PASS** |
| TC-07     | Low Escalation Target               | Math Boundary $< 0.70 \to \text{False}$  | **PASS** |
| TC-08     | Email Address Scrubbing             | Regex PII Masking (`[EMAIL]`)            | **PASS** |
| TC-09     | Phone Number Scrubbing              | Regex PII Masking (`[PHONE]`)            | **PASS** |
| TC-10     | Basic Instruction Override          | Prompt Injection Interception            | **PASS** |
| TC-11     | Jailbreak Attack Pattern            | Security Rejection & Auto-Escalate       | **PASS** |
| TC-12     | Out-of-Domain / Recipe Query        | Grounded Refusal Generation              | **PASS** |
| TC-13     | Multi-turn Session Memory           | Prior Dialog State Retention             | **PASS** |
| TC-14     | Grounded Response Caching           | Normalized Key Cache Hits                | **PASS** |
| TC-15     | Runtime Token Budget Cap            | BudgetExceededException Catch            | **PASS** |

### Stress & Concurrency Suite (`scaling_eval.py`)

Execution against `scaling_eval.py` confirms stability under burst load and rapid cache delivery:

```
============================================================
🚀 Running Naukri Agent Scaling & Stress Evaluation Suite
============================================================

[1/3] Runtime Budget Verification...
  - Request within budget accepted: True
  - Request exceeding budget intercepted: True

[2/3] Cache Latency & Speedup Benchmark...
  - Cold RAG + Crew Latency: 842.15 ms
  - Warm Cache Hit Latency:  0.0032 ms
  - Latency Speedup:          263171.9x

[3/3] Burst Concurrency Benchmark (10 Workers)...
  - Success Rate: 100.0%
  - Mean Latency: 859.34 ms
  - P50 Latency:  858.21 ms
  - P95 Latency:  874.12 ms

============================================================
✅ All Scaling and Stress Verification Checks PASSED.
============================================================
```

---

## 8. Setup, Installation & Verification

### Prerequisites

- Python 3.10+
- Virtual environment tool (`venv` or `conda`)

### Installation Steps

1. **Clone the repository and enter directory:**

   ```bash
   git clone https://github.com/organization/naukri-domain-support-agent.git
   cd naukri-domain-support-agent
   ```

2. **Configure virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Verify Environment Configurations (`.env`):**
   Ensure `.env` contains the required parameters:

   ```ini
   LLM_MODE=mock
   MAX_TOKEN_BUDGET_PER_REQUEST=350
   ESCALATION_THRESHOLD=0.70
   AUDIT_LOG_PATH=agent_audit.jsonl
   CREWAI_DISABLE_TELEMETRY=true
   OTEL_SDK_DISABLED=true
   CREWAI_TRACING_ENABLED=false
   HF_HUB_DISABLE_SYMLINKS_WARNING=1
   ```

4. **Run Functional Verification Suite:**

   ```bash
   python evaluate.py
   ```

5. **Run Scaling, Concurrency & Budget Suite:**
   ```bash
   python scaling_eval.py
   ```
