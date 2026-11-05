import os
import re
from typing import Union, List, Dict, Any, Optional
from crewai.llms.base_llm import BaseLLM
from tools import check_job_application_status
from rag_engine import grounded_rag_query

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TRACING_ENABLED"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Shared context so LLM calls always resolve the true active query
_ACTIVE_QUERY_CONTEXT: Dict[str, Any] = {
    "raw_query": "",
    "clean_query": "",
    "app_id": None,
    "is_lookup": False,
}

def set_active_query_context(query: str):
    """Stores the active user query for deterministic resolution."""
    raw = query.strip()
    clean = re.sub(r'\(Prior context:.*?\)', '', raw, flags=re.IGNORECASE | re.DOTALL).strip()
    clean = clean.strip("'\"").strip()

    app_match = re.search(r'\b(APP-\d{3,5})\b', raw, re.IGNORECASE)
    app_id = app_match.group(1).upper() if app_match else None

    _ACTIVE_QUERY_CONTEXT["raw_query"] = raw
    _ACTIVE_QUERY_CONTEXT["clean_query"] = clean
    _ACTIVE_QUERY_CONTEXT["app_id"] = app_id
    _ACTIVE_QUERY_CONTEXT["is_lookup"] = app_id is not None

def get_active_query_context() -> Dict[str, Any]:
    return _ACTIVE_QUERY_CONTEXT

def extract_core_query(text: str) -> str:
    """Extract clean query from text, falling back to active query context."""
    ctx = get_active_query_context()
    if ctx.get("clean_query"):
        return ctx["clean_query"]

    if not text:
        return ""

    m = re.search(r"query:\s*['\"]+(.*?)['\"]+(?:\s+using|\.|\n|$)", text, re.IGNORECASE | re.DOTALL)
    if m:
        candidate = m.group(1).strip()
    else:
        m_user = re.search(r"(?:User|Human|Task Description):\s*([^\n]+)", text, re.IGNORECASE)
        candidate = m_user.group(1).strip() if m_user else text.strip()

    candidate = re.sub(r'\(Prior context:.*?\)', '', candidate, flags=re.IGNORECASE | re.DOTALL)
    candidate = re.sub(r'^(?:Retrieve policy details for|Lookup record details for|query:)\s*', '', candidate, flags=re.IGNORECASE)
    return candidate.strip().strip("'\"").strip()

class DeterministicMockLLM(BaseLLM):
    def __init__(self, model: str = "deterministic-mock", temperature: Optional[float] = 0.0):
        super().__init__(model=model, temperature=temperature)

    def call(
        self,
        messages: Union[str, List[Dict[str, str]], List[Any]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        ctx = get_active_query_context()
        app_id = ctx.get("app_id")
        clean_q = ctx.get("clean_query", "")

        # Flatten message text
        if isinstance(messages, str):
            full_text = messages
        elif isinstance(messages, list):
            parts = []
            for m in messages:
                if isinstance(m, dict):
                    parts.append(m.get("content", ""))
                elif hasattr(m, "content"):
                    parts.append(str(m.content))
                else:
                    parts.append(str(m))
            full_text = "\n".join(parts)
        else:
            full_text = str(messages)

        # Fallback check on text if context wasn't initialized
        if not app_id:
            app_match = re.search(r'\b(APP-\d{3,5})\b', full_text, re.IGNORECASE)
            if app_match:
                app_id = app_match.group(1).upper()

        if not clean_q:
            clean_q = extract_core_query(full_text)

        is_composer = "Response Composer" in full_text
        has_observation = "Observation:" in full_text

        # -------------------------------------------------------------
        # 1. Worker Agent Initial Action
        # -------------------------------------------------------------
        if not is_composer and not has_observation:
            if app_id:
                return (
                    f"Thought: Looking up candidate application {app_id}.\n"
                    f"Action: check_job_application_status\n"
                    f"Action Input: {{\"record_id\": \"{app_id}\"}}"
                )
            else:
                return (
                    f"Thought: Searching policy knowledge base for relevant clauses.\n"
                    f"Action: rag_lookup\n"
                    f"Action Input: {{\"query\": \"{clean_q}\"}}"
                )

        # -------------------------------------------------------------
        # 2. Worker Agent Finishing on Observation
        # -------------------------------------------------------------
        if not is_composer and has_observation:
            obs = re.findall(r'Observation\s*:\s*(.*?)(?=\nThought:|\nAction:|\Z)', full_text, re.DOTALL | re.IGNORECASE)
            last_obs = obs[-1].strip() if obs else "Done."
            return f"Thought: I have retrieved the required data.\nFinal Answer: {last_obs}"

        # -------------------------------------------------------------
        # 3. Response Composer (Synthesis)
        # -------------------------------------------------------------
        if is_composer:
            # Case A: Candidate Application Lookup
            if app_id:
                lookup_res = check_job_application_status(app_id)
                if not lookup_res.get("found", False):
                    return (
                        f"Thought: Application record was not found.\n"
                        f"Final Answer: Application record {app_id} was not found in the database. "
                        f"Details: {lookup_res}"
                    )
                return (
                    f"Thought: Application record found.\n"
                    f"Final Answer: Application status details for {app_id}: {lookup_res}"
                )

            # Case B: Out-of-Domain Guard
            out_of_domain = ["cookie", "recipe", "weather", "capital of", "president", "sports score"]
            if any(term in clean_q.lower() for term in out_of_domain):
                return (
                    "Thought: Query falls outside domain.\n"
                    "Final Answer: I do not have sufficient information in the knowledge base to answer this query."
                )

            # Case C: Context passed from Worker Task
            obs_matches = re.findall(
                r'(?:Observation|Context from prior tasks:)\s*:\s*(.*?)(?=\nThought:|\nAction:|\Z)',
                full_text,
                re.DOTALL | re.IGNORECASE
            )
            for block in reversed(obs_matches):
                block_clean = block.strip()
                if "According to Naukri policy" in block_clean:
                    return f"Thought: Synthesizing policy from context.\nFinal Answer: {block_clean}"

            # Case D: Direct Grounded Query Fallback
            rag_res = grounded_rag_query(clean_q, collection_name="sentence_chunk_kb")
            ans = rag_res.get("answer", "")
            if not ans.startswith("According to Naukri policy") and "insufficient information" not in ans.lower():
                ans = f"According to Naukri policy: {ans}"
            return f"Thought: Formulating verified policy response.\nFinal Answer: {ans}"

        return "Thought: Done.\nFinal Answer: Request completed."

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return True

    def get_context_window_size(self) -> int:
        return 8192