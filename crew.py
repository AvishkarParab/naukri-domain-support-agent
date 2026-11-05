"""
crew.py - CrewAI Multi-Agent System with Least Autonomy, Memory, and Guardrails (Tasks 7, 8, 9, 10, 15)
"""

import os
from typing import Dict, Any, List

# Suppress telemetry and interactive prompts
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TRACING_ENABLED"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from mock_llm import DeterministicMockLLM
from tools import check_job_application_status
from rag_engine import grounded_rag_query
from guardrails import (
    mask_pii,
    detect_prompt_injection,
    verify_groundedness,
    AgentResponseSchema,
)


# --- Wrap Tools with CrewAI Decorator ---

@tool("rag_lookup")
def rag_lookup_tool(query: str) -> str:
    """Searches the official Naukri HR policy knowledge base for relevant clauses."""
    res = grounded_rag_query(query, collection_name="sentence_chunk_kb")
    return res["answer"]


@tool("check_job_application_status")
def application_lookup_tool(record_id: str) -> str:
    """Looks up a candidate application record, salary, active days, and escalation metrics."""
    res = check_job_application_status(record_id)
    return str(res)


# --- Task 8: Modern In-Memory Multi-Turn Session Store ---

class InMemoryChatHistory(BaseChatMessageHistory):
    """Modern, non-deprecated in-memory chat message history."""
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
    """Retrieves or creates an in-memory conversation history for a given session."""
    if session_id not in SESSION_STORE:
        SESSION_STORE[session_id] = InMemoryChatHistory()
    return SESSION_STORE[session_id]


def reset_session_history(session_id: str):
    """Resets memory state for a specific session."""
    if session_id in SESSION_STORE:
        del SESSION_STORE[session_id]


# --- Task 7 & 15: Agent Crew Construction with Least Autonomy ---

def build_naukri_crew(is_lookup: bool) -> Crew:
    mock_llm = DeterministicMockLLM()

    # Agent 1: Retrieval Agent (Allowed ONLY rag_lookup)
    retrieval_agent = Agent(
        role="Retrieval Agent",
        goal="Retrieve accurate policy clauses from the knowledge base for inquiries.",
        backstory="An expert on official Naukri HR regulations, documentation, and compliance guidelines.",
        tools=[rag_lookup_tool],
        llm=mock_llm,
        verbose=False,
    )

    # Agent 2: Lookup Agent (Allowed ONLY check_job_application_status - Least Autonomy)
    lookup_agent = Agent(
        role="Lookup Agent",
        goal="Lookup candidate application status and report calculated escalation scores.",
        backstory="A candidate database specialist strictly authorized to query recruitment records.",
        tools=[application_lookup_tool],
        llm=mock_llm,
        verbose=False,
    )

    # Agent 3: Response Composer (Strictly 0 tools - Synthesizer only)
    composer_agent = Agent(
        role="Response Composer",
        goal="Synthesize retrieved data and policy clauses into a clear, grounded response.",
        backstory="A senior support specialist dedicated to producing grounded, empathetic, and compliant support answers.",
        tools=[],  # Strictly 0 tools: Least Autonomy enforced
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
        description="Review the retrieved information from the previous task and draft a clear final answer.",
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


# --- End-to-End Orchestrated Pipeline (Tasks 7, 8, 9, 10) ---

def run_orchestrated_query(query: str, session_id: str = "default_session") -> AgentResponseSchema:
    # 1. Input Guardrail: PII Masking
    sanitized_query = mask_pii(query)

    # 2. Input Guardrail: Prompt Injection Detection
    if detect_prompt_injection(sanitized_query):
        return AgentResponseSchema(
            query=sanitized_query,
            status="refused",
            answer="Security policy violation: prompt injection attempt detected.",
            sources=[],
            escalation_flag=True,
        )

    # 3. Retrieve Session Memory Context
    history = get_session_history(session_id)
    history.add_user_message(sanitized_query)

    recent_messages = history.messages[-4:]
    history_context = " | ".join([f"{m.type}: {m.content}" for m in recent_messages[:-1]])

    if history_context:
        enriched_query = f"{sanitized_query} (Prior context: {history_context})"
    else:
        enriched_query = sanitized_query

    is_lookup = "APP-" in sanitized_query.upper()

    # 4. Multi-Agent Crew Execution
    crew = build_naukri_crew(is_lookup=is_lookup)
    raw_output = crew.kickoff(inputs={"user_query": enriched_query})
    raw_answer = str(raw_output).strip()

    # Identify source references
    sources: List[str] = []
    if is_lookup:
        sources.append("JOB_APPLICATIONS_DB")
    if "According to Naukri policy" in raw_answer or "sentence_chunk_kb" in raw_answer:
        sources.append("sentence_chunk_kb")

    # 5. Output Guardrail: Groundedness Check
    is_grounded = verify_groundedness(raw_answer, sources)
    status_str = "success" if is_grounded else "refused"

    if not is_grounded and "insufficient information" in raw_answer.lower():
        status_str = "refused"

    escalation_triggered = "recommend_escalation': True" in raw_answer or "recommend_escalation=True" in raw_answer

    # 6. Commit assistant response to session memory
    history.add_ai_message(raw_answer)

    # 7. Validate and return structured Pydantic schema (Task 9)
    return AgentResponseSchema(
        query=sanitized_query,
        status=status_str,
        answer=raw_answer,
        sources=sources,
        escalation_flag=escalation_triggered,
    )


if __name__ == "__main__":
    print("=== STARTING CREW VERIFICATION ===", flush=True)

    print("\n--- 1. TESTING RAG QUERY ---", flush=True)
    rag_test = run_orchestrated_query("What is the standard notice period for permanent staff?")
    print(f"Status : {rag_test.status}")
    print(f"Answer : {rag_test.answer}")
    print(f"Sources: {rag_test.sources}")

    print("\n--- 2. TESTING LOOKUP QUERY ---", flush=True)
    lookup_test = run_orchestrated_query("Can you check the application status for APP-1002?")
    print(f"Status     : {lookup_test.status}")
    print(f"Answer     : {lookup_test.answer}")
    print(f"Escalation : {lookup_test.escalation_flag}")

    print("\n=== CREW VERIFICATION COMPLETE ===", flush=True)