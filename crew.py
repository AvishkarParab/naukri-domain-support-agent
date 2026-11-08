import os
import re
from typing import Dict, List
import asyncio
import concurrent.futures
from review_team import run_review_stage

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TRACING_ENABLED"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from mock_llm import (
    DeterministicMockLLM,
    extract_core_query,
    set_active_query_context,
)
from tools import check_job_application_status
from rag_engine import grounded_rag_query
from guardrails import (
    mask_pii,
    detect_prompt_injection,
    verify_groundedness,
    AgentResponseSchema,
)


@tool("rag_lookup")
def rag_lookup_tool(query: str) -> str:
    """Searches official Naukri HR policy knowledge base for relevant clauses."""
    clean_q = extract_core_query(query)
    res = grounded_rag_query(clean_q, collection_name="sentence_chunk_kb")
    ans = res.get("answer", "")
    if not ans.startswith("According to Naukri policy") and "insufficient information" not in ans.lower():
        ans = f"According to Naukri policy: {ans}"
    return ans


@tool("check_job_application_status")
def application_lookup_tool(record_id: str) -> str:
    """Looks up candidate application record, salary, active days, and escalation metrics."""
    clean_id = re.search(r'\b(APP-\d{3,5})\b', str(record_id), re.IGNORECASE)
    rid = clean_id.group(1).upper() if clean_id else str(record_id).strip()
    res = check_job_application_status(rid)
    return str(res)


class InMemoryChatHistory(BaseChatMessageHistory):
    def __init__(self) -> None:
        self.messages: List[BaseMessage] = []

    def add_messages(self, messages: List[BaseMessage]) -> None:
        self.messages.extend(messages)

    def add_user_message(self, message: str) -> None:
        self.messages.append(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        self.messages.append(AIMessage(content=message))

    def clear(self) -> None:
        self.messages.clear()


SESSION_STORE: Dict[str, InMemoryChatHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatHistory:
    if session_id not in SESSION_STORE:
        SESSION_STORE[session_id] = InMemoryChatHistory()
    return SESSION_STORE[session_id]


def reset_session_history(session_id: str):
    if session_id in SESSION_STORE:
        del SESSION_STORE[session_id]


def build_naukri_crew(is_lookup: bool) -> Crew:
    mock_llm = DeterministicMockLLM()

    retrieval_agent = Agent(
        role="Retrieval Agent",
        goal="Retrieve accurate policy clauses from the knowledge base for inquiries.",
        backstory="An expert on official Naukri HR regulations, documentation, and compliance guidelines.",
        tools=[rag_lookup_tool],
        llm=mock_llm,
        verbose=False,
    )

    lookup_agent = Agent(
        role="Lookup Agent",
        goal="Lookup candidate application status and report calculated escalation scores.",
        backstory="A candidate database specialist strictly authorized to query recruitment records.",
        tools=[application_lookup_tool],
        llm=mock_llm,
        verbose=False,
    )

    composer_agent = Agent(
        role="Response Composer",
        goal="Synthesize retrieved data and policy clauses into a clear, grounded response.",
        backstory="A senior support specialist dedicated to producing grounded, empathetic, and compliant support answers.",
        tools=[],  # Strictly 0 tools
        llm=mock_llm,
        verbose=False,
    )

    if is_lookup:
        worker_agent = lookup_agent
        worker_task_desc = "Lookup record details for query: '{user_query}' using the application lookup tool."
    else:
        worker_agent = retrieval_agent
        worker_task_desc = "Retrieve policy details for query: '{user_query}' using the RAG lookup tool."

    task_fetch = Task(
        description=worker_task_desc,
        expected_output="Raw data or policy information retrieved via the authorized tool.",
        agent=worker_agent,
    )

    task_compose = Task(
        description="Review the retrieved information from the previous task for query: '{user_query}' and draft a clear final answer.",
        expected_output="A grounded, structured final response addressing the inquiry.",
        agent=composer_agent,
        context=[task_fetch],
    )

    return Crew(
        agents=[retrieval_agent, lookup_agent, composer_agent],
        tasks=[task_fetch, task_compose],
        process=Process.sequential,
        verbose=False,
    )

def run_async_safely(coro):
    """
    Safely executes an async coroutine from synchronous code,
    preventing 'RuntimeError: This event loop is already running'
    when invoked inside FastAPI or active asyncio event loops.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)

def run_orchestrated_query(query: str, session_id: str = "default_session") -> AgentResponseSchema:
    sanitized_query = mask_pii(query)

    if detect_prompt_injection(sanitized_query):
        return AgentResponseSchema(
            query=sanitized_query,
            status="refused",
            answer="Security policy violation: prompt injection attempt detected.",
            sources=[],
            escalation_flag=True,
        )

    history = get_session_history(session_id)
    history.add_user_message(sanitized_query)

    recent_messages = history.messages[-4:]
    history_context = " | ".join([f"{m.type}: {m.content}" for m in recent_messages[:-1]])

    if history_context:
        enriched_query = f"{sanitized_query} (Prior context: {history_context})"
    else:
        enriched_query = sanitized_query

    # Set active query context for DeterministicMockLLM
    set_active_query_context(enriched_query)

    is_lookup = bool(re.search(r'\b(APP-\d{3,5})\b', enriched_query, re.IGNORECASE))

    crew = build_naukri_crew(is_lookup=is_lookup)
    raw_output = crew.kickoff(inputs={"user_query": enriched_query})
    composer_draft = str(raw_output).strip()

    if not is_lookup and "insufficient information" not in composer_draft.lower():
        try:
            verdict = run_async_safely(run_review_stage(composer_draft, enriched_query))
            raw_answer = verdict.final_answer
        except Exception:
            raw_answer = composer_draft
    else:
        raw_answer = composer_draft

    sources: List[str] = []
    if is_lookup or "APP-" in raw_answer:
        sources.append("JOB_APPLICATIONS_DB")
    if "According to Naukri policy" in raw_answer or "sentence_chunk_kb" in raw_answer:
        sources.append("sentence_chunk_kb")

    is_grounded = verify_groundedness(raw_answer, sources)

    if "insufficient information" in raw_answer.lower() or not is_grounded:
        status_str = "refused"
    else:
        status_str = "success"

    escalation_triggered = (
        "recommend_escalation': True" in raw_answer
        or "recommend_escalation=True" in raw_answer
        or "'recommend_escalation': True" in raw_answer
    )

    history.add_ai_message(raw_answer)

    return AgentResponseSchema(
        query=sanitized_query,
        status=status_str,
        answer=raw_answer,
        sources=sources,
        escalation_flag=escalation_triggered,
    )