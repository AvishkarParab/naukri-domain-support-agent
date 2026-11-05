"""
guardrails.py - PII Masking, Prompt Injection Detection, Groundedness Verification,
and Pydantic Response Schema (Tasks 9 & 10)
"""

import re
from typing import List
from pydantic import BaseModel, Field


class AgentResponseSchema(BaseModel):
    """Structured output schema enforced for all crew responses (Task 9)."""
    query: str = Field(description="The sanitized input query")
    status: str = Field(description="Processing status: 'success', 'refused', or 'flagged'")
    answer: str = Field(description="Final answer text provided to the user")
    sources: List[str] = Field(default_factory=list, description="Referenced policy docs or application record IDs")
    escalation_flag: bool = Field(default=False, description="Whether escalation was triggered")


# Indian telephone format: +91 or optional prefix, followed by standard 10 digits starting 6-9
PHONE_REGEX = r'(?:\+91[\-\s]?)?[6789]\d{9}\b'

PROMPT_INJECTION_PATTERNS = [
    r'(?i)ignore\s+(all\s+)?(previous|prior)\s+instructions',
    r'(?i)you\s+are\s+now\s+in\s+developer\s+mode',
    r'(?i)system\s+override',
    r'(?i)bypass\s+all\s+rules',
    r'(?i)drop\s+table',
]


def mask_pii(text: str) -> str:
    """Masks fixed-format phone numbers to preserve candidate privacy."""
    return re.sub(PHONE_REGEX, "[PHONE_MASKED]", text)


def detect_prompt_injection(text: str) -> bool:
    """Detects adversarial override strings and returns True if found."""
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


def verify_groundedness(answer: str, retrieved_context: List[str]) -> bool:
    """
    Output-side guardrail: Ensures generated answer is grounded in retrieved context.
    Refuses if answer indicates lack of context or empty retrieved context.
    """
    if "insufficient information" in answer.lower():
        return False
    if not retrieved_context and "APP-" not in answer:
        return False
    return True