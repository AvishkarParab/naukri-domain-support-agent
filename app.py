"""
app.py - Production-Ready FastAPI Deployment, ELK-Style Logging & WebSockets (Tasks 11 & 12)
"""

import time
import uuid
import json
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel, Field

from crew import run_orchestrated_query
from rag_engine import sentence_collection
from guardrails import mask_pii, AgentResponseSchema
from cache import RESPONSE_CACHE, enforce_runtime_budget, BudgetExceededException

import os
from dotenv import load_dotenv

load_dotenv()

# Configure ELK-style JSON-Lines Structured Logger (Task 12)
logger = logging.getLogger("naukri_audit")
logger.setLevel(logging.INFO)

AUDIT_LOG_PATH = os.getenv("AUDIT_LOG_PATH", "agent_audit.jsonl")

log_handler = logging.FileHandler(AUDIT_LOG_PATH, encoding="utf-8")
log_handler.setFormatter(logging.Formatter('%(message)s'))
if not logger.handlers:
    logger.addHandler(log_handler)


def log_structured_event(trace_id: str, query: str, response: Dict[str, Any], duration_ms: float, cached: bool = False):
    """
    Emits an ELK-compatible JSON-Lines log entry.
    Enforces that unmasked fixed-format PII NEVER reaches disk in the clear.
    """
    safe_query = mask_pii(query)
    record = {
        "timestamp": time.time(),
        "trace_id": trace_id,
        "query": safe_query,
        "response": response,
        "duration_ms": round(duration_ms, 2),
        "cached": cached,
    }
    logger.info(json.dumps(record))


app = FastAPI(
    title="Naukri Domain Support Agent API",
    version="1.0.0",
    description="Production-grade domain support API with CrewAI orchestration, AutoGen review, and ChromaDB RAG."
)


# --- Pydantic Request Models ---

class AskRequest(BaseModel):
    query: str = Field(description="Recruiter or candidate support query")
    session_id: str = Field(default="default_session", description="Session identifier for multi-turn history")


class AddDocumentRequest(BaseModel):
    doc_id: str = Field(description="Unique document identifier")
    topic: str = Field(description="Policy domain or topic category")
    content: str = Field(description="Textual policy content to index (2-5 sentences)")


class AddDocumentResponse(BaseModel):
    status: str
    doc_id: str
    message: str


# --- HTTP Endpoints ---

@app.post("/ask", response_model=AgentResponseSchema)
def ask_endpoint(req: AskRequest):
    """
    Primary support endpoint: Checks budget, checks cache, runs crew orchestration,
    and logs structured audit trails with masked PII.
    """
    trace_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    # 1. Task 15: Runtime Budget Cap
    try:
        enforce_runtime_budget(req.query)
    except BudgetExceededException as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2. Task 16: Check Normalized Response Cache
    cached_answer = RESPONSE_CACHE.get(req.query)
    if cached_answer:
        duration_ms = (time.perf_counter() - start_time) * 1000
        response_obj = AgentResponseSchema(
            query=mask_pii(req.query),
            status="success",
            answer=f"[CACHE HIT] {cached_answer}",
            sources=["RESPONSE_CACHE"],
            escalation_flag=False,
        )
        log_structured_event(trace_id, req.query, response_obj.model_dump(), duration_ms, cached=True)
        return response_obj

    # 3. Task 7-10: Multi-Agent Crew Execution
    result = run_orchestrated_query(req.query, session_id=req.session_id)

    # 4. Save to Cache if grounded success
    if result.status == "success":
        RESPONSE_CACHE.set(req.query, result.answer)

    duration_ms = (time.perf_counter() - start_time) * 1000
    log_structured_event(trace_id, req.query, result.model_dump(), duration_ms, cached=False)
    return result


@app.post("/add-document", response_model=AddDocumentResponse)
def add_document_endpoint(req: AddDocumentRequest):
    """
    Dynamic document ingestion: Upserts new policy document into ChromaDB sentence collection.
    """
    from rag_engine import chunk_sentence_based

    chunks = chunk_sentence_based(req.content)
    chunk_ids = [f"{req.doc_id}_dyn_{i}" for i in range(len(chunks))]
    metadatas = [{"parent_id": req.doc_id, "topic": req.topic} for _ in chunks]

    sentence_collection.upsert(ids=chunk_ids, documents=chunks, metadatas=metadatas)

    return AddDocumentResponse(
        status="success",
        doc_id=req.doc_id,
        message=f"Document successfully indexed into {len(chunks)} sentence chunks."
    )


# --- WebSocket Endpoint (Task 11) ---

@app.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time multi-turn conversation.
    Gracefully catches WebSocketDisconnect to keep the server operational for others.
    """
    await websocket.accept()
    session_id = f"ws_{uuid.uuid4().hex[:8]}"

    try:
        while True:
            raw_text = await websocket.receive_text()
            trace_id = str(uuid.uuid4())
            start_time = time.perf_counter()

            # Enforce budget guard
            try:
                enforce_runtime_budget(raw_text)
            except BudgetExceededException as e:
                await websocket.send_text(json.dumps({
                    "error": str(e),
                    "status": "budget_exceeded"
                }))
                continue

            # Process turn with session continuity
            result = run_orchestrated_query(raw_text, session_id=session_id)
            duration_ms = (time.perf_counter() - start_time) * 1000

            log_structured_event(trace_id, raw_text, result.model_dump(), duration_ms)
            await websocket.send_text(json.dumps(result.model_dump()))

    except WebSocketDisconnect:
        # Graceful handling without crashing server process
        print(f"[WebSocket] Client disconnected cleanly: session_id={session_id}")