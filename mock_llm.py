"""
mock_llm.py - Deterministic, Offline BaseLLM implementation for CrewAI
"""

import os
import re
from typing import Union, List, Dict, Any, Optional
from crewai.llms.base_llm import BaseLLM

# Zero-network enforcement
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TRACING_ENABLED"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


class DeterministicMockLLM(BaseLLM):
    """
    Offline Mock LLM satisfying CrewAI's BaseLLM interface without external network or API keys.
    """
    def __init__(self, model: str = "deterministic-mock", temperature: Optional[float] = 0.0):
        super().__init__(model=model, temperature=temperature)

    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        if isinstance(messages, str):
            full_text = messages
        elif isinstance(messages, list):
            full_text = "\n".join([
                m.get("content", "") if isinstance(m, dict) else str(m)
                for m in messages
            ])
        else:
            full_text = str(messages)

        is_composer = "Response Composer" in full_text
        is_lookup = "Lookup Agent" in full_text or "check_job_application_status" in full_text
        is_retrieval = "Retrieval Agent" in full_text or "rag_lookup" in full_text

        has_real_observation = bool(re.search(r'Action\s*:\s*\w+.*?Observation\s*:', full_text, re.DOTALL))

        # -------------------------------------------------------------
        # CASE 1: WORKER AGENT EXECUTIONS (Task 1)
        # -------------------------------------------------------------
        if is_lookup and not has_real_observation and not is_composer:
            match = re.search(r'\b(APP-\d{4})\b', full_text, re.IGNORECASE)
            record_id = match.group(1).upper() if match else "APP-1002"
            return (
                f"Thought: I need to look up application details for {record_id}.\n"
                f"Action: check_job_application_status\n"
                f"Action Input: {{\"record_id\": \"{record_id}\"}}"
            )

        if is_retrieval and not has_real_observation and not is_composer:
            query_match = re.search(r'query:\s*[\'"]?([^\'"}\n]+)', full_text, re.IGNORECASE)
            extracted_query = query_match.group(1).strip() if query_match else "notice period policy"
            return (
                f"Thought: I need to retrieve official Naukri HR policy clauses.\n"
                f"Action: rag_lookup\n"
                f"Action Input: {{\"query\": \"{extracted_query}\"}}"
            )

        if has_real_observation and not is_composer:
            observations = re.findall(r'Observation\s*:\s*(.*?)(?=\nThought:|\nAction:|\Z)', full_text, re.DOTALL)
            last_obs = observations[-1].strip() if observations else "Record retrieved."
            return (
                f"Thought: Tool executed successfully.\n"
                f"Final Answer: {last_obs}"
            )

        # -------------------------------------------------------------
        # CASE 2: RESPONSE COMPOSER (Task 2)
        # -------------------------------------------------------------
        if is_composer:
            # If this is an application inquiry, prioritize the application record
            if "APP-" in full_text.upper():
                dict_match = re.search(r'(\{.*?\'status\'.*?\})', full_text, re.DOTALL)
                if dict_match:
                    app_info = dict_match.group(1)
                else:
                    match_id = re.search(r'\b(APP-\d{4})\b', full_text, re.IGNORECASE)
                    app_id = match_id.group(1).upper() if match_id else "APP-1002"
                    from tools import check_job_application_status
                    app_info = str(check_job_application_status(app_id))

                return (
                    f"Thought: Candidate application record retrieved and verified.\n"
                    f"Final Answer: Application record details: {app_info}"
                )

            # Otherwise, handle policy retrieval
            if "According to Naukri policy" in full_text:
                policy_match = re.search(r'(According to Naukri policy[^\n]+)', full_text)
                policy_sentence = policy_match.group(1) if policy_match else (
                    "According to Naukri policy on notice-period policy: Standard employment contracts "
                    "stipulate a mandatory ninety-day notice period prior to voluntary release."
                )
                return (
                    f"Thought: Synthesizing policy clauses.\n"
                    f"Final Answer: {policy_sentence}"
                )

            return (
                "Thought: Synthesizing final answer.\n"
                "Final Answer: According to Naukri policy on notice-period policy: Standard employment contracts "
                "stipulate a mandatory ninety-day notice period prior to voluntary release."
            )

        return (
            "Thought: Processing request.\n"
            "Final Answer: Inquiry processed according to Naukri support standards."
        )

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return True

    def get_context_window_size(self) -> int:
        return 8192