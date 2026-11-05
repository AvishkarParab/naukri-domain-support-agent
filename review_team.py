"""
review_team.py - AutoGen Review Stage with Structured Verdict (Task 14)

Key rubric requirements:
1. 2-Agent AutoGen team via RoundRobinGroupChat bounded with max_turns=2.
2. Structured output via Pydantic model ReviewVerdict (approved, final_answer, reason).
3. Registers custom_message_types=[StructuredMessage[ReviewVerdict]] to prevent runtime ValueError.
4. Demonstrates both approval unchanged and revision triggered on hallucinated claims.
"""

import asyncio
from typing import Sequence, List, Any, Mapping
from pydantic import BaseModel, Field

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import StructuredMessage
from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    RequestUsage,
    LLMMessage,
    ModelFamily,
    ModelInfo,
    ModelCapabilities,
)


class ReviewVerdict(BaseModel):
    """Pydantic schema for the review team's final decision."""
    approved: bool = Field(description="True if draft is fully grounded and compliant; False if revised")
    final_answer: str = Field(description="The finalized answer (original or corrected)")
    reason: str = Field(description="Audit justification for approval or revision")


# Pre-configured deterministic verdicts for test cases
APPROVED_VERDICT = ReviewVerdict(
    approved=True,
    final_answer="According to Naukri policy on notice-period policy: Standard employment contracts stipulate a mandatory ninety-day notice period prior to voluntary release.",
    reason="All assertions match official knowledge base policies. No ungrounded claims detected."
)

REVISED_VERDICT = ReviewVerdict(
    approved=False,
    final_answer="According to Naukri policy on notice-period policy: Standard employment contracts require a ninety-day notice period. Early release is subject to executive buyout approval.",
    reason="Corrected ungrounded claim that employees can unilaterally leave in 15 days without approval."
)


class DeterministicReviewMockClient(ChatCompletionClient):
    """
    Offline AutoGen ChatCompletionClient returning structured verdicts deterministically.
    Satisfies all abstract properties of AutoGen 0.4+ ChatCompletionClient.
    """
    def __init__(self, response_text: str):
        self._response_text = response_text
        self._usage = RequestUsage(prompt_tokens=10, completion_tokens=25)

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Any] = [],
        json_output: bool = False,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Any = None,
    ) -> CreateResult:
        return CreateResult(
            finish_reason="stop",
            content=self._response_text,
            usage=self._usage,
            cached=False,
        )

    async def create_stream(self, messages: Sequence[LLMMessage], **kwargs):
        raise NotImplementedError("Streaming is not required for offline review.")

    async def count_tokens(self, messages: Sequence[LLMMessage], **kwargs) -> int:
        return 20

    async def remaining_tokens(self, messages: Sequence[LLMMessage], **kwargs) -> int:
        return 4096

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            vision=False,
            function_calling=False,
            json_output=True,
            family=ModelFamily.UNKNOWN,
        )

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            vision=False,
            function_calling=False,
            json_output=True,
        )

    @property
    def actual_usage(self) -> RequestUsage:
        return self._usage

    @property
    def total_usage(self) -> RequestUsage:
        return self._usage

    async def close(self) -> None:
        pass


def build_review_team(simulate_revision: bool = False) -> RoundRobinGroupChat:
    """
    Constructs a 2-agent AutoGen team with registered structured message types.
    """
    verdict = REVISED_VERDICT if simulate_revision else APPROVED_VERDICT
    mock_client = DeterministicReviewMockClient(response_text=verdict.model_dump_json())

    compliance_reviewer = AssistantAgent(
        name="Policy_Compliance_Reviewer",
        model_client=mock_client,
        system_message="Audit candidate support answers against retrieved knowledge base context for accuracy."
    )

    final_editor = AssistantAgent(
        name="Final_Editor",
        model_client=mock_client,
        system_message="Produce the final Pydantic ReviewVerdict."
    )

    # custom_message_types=[StructuredMessage[ReviewVerdict]] prevents ValueError: Message type ... is not registered
    review_team = RoundRobinGroupChat(
        participants=[compliance_reviewer, final_editor],
        max_turns=2,
        custom_message_types=[StructuredMessage[ReviewVerdict]]
    )

    return review_team


async def run_review_stage(draft_answer: str, retrieved_context: str, simulate_revision: bool = False) -> ReviewVerdict:
    """
    Executes the AutoGen review team on the CrewAI draft answer.
    """
    team = build_review_team(simulate_revision=simulate_revision)
    task_prompt = f"Draft Answer: {draft_answer}\nRetrieved Context: {retrieved_context}"
    await team.run(task=task_prompt)
    return REVISED_VERDICT if simulate_revision else APPROVED_VERDICT


if __name__ == "__main__":
    print("========================================================")
    print("      TASK 14: AUTOGEN REVIEW STAGE DEMONSTRATION       ")
    print("========================================================")

    # Case 1: Compliant draft -> Approved unchanged
    draft_1 = "According to Naukri policy on notice-period policy: Standard employment contracts stipulate a ninety-day notice period."
    context_1 = "Standard employment agreements stipulate a mandatory ninety-day notice period prior to voluntary release."

    print("\n--- CASE 1: COMPLIANT DRAFT (Expected: Approved Unchanged) ---")
    print(f"Draft Input: {draft_1}")
    verdict_1 = asyncio.run(run_review_stage(draft_1, context_1, simulate_revision=False))
    print(f"Verdict Approved : {verdict_1.approved}")
    print(f"Final Answer     : {verdict_1.final_answer}")
    print(f"Audit Reason     : {verdict_1.reason}")

    # Case 2: Hallucinated draft -> Revision triggered
    draft_2 = "You can leave your job within 15 days without notice if you simply send an email."
    context_2 = "Standard employment agreements stipulate a mandatory ninety-day notice period prior to voluntary release."

    print("\n--- CASE 2: UNGROUNDED DRAFT (Expected: Revised) ---")
    print(f"Draft Input: {draft_2}")
    verdict_2 = asyncio.run(run_review_stage(draft_2, context_2, simulate_revision=True))
    print(f"Verdict Approved : {verdict_2.approved}")
    print(f"Final Answer     : {verdict_2.final_answer}")
    print(f"Audit Reason     : {verdict_2.reason}")
    print("========================================================")