"""
llm — Protocol-based LLM interaction layer.

Protocols
---------
CallLLM     Send a prompt with context to an LLM, get structured response back.

Implementations
---------------
call_llm_stub       Deterministic stub (no LLM dependency).
call_llm_openai     Real OpenAI implementation (requires API key).

Validation
----------
LLMValidator        Pre-run quiz to verify LLM domain understanding.
ValidationQuiz      A single quiz question with required/prohibited concepts.

Swap in any callable that matches the Protocol signature.
"""

from conversation_engine.infrastructure.llm.protocols import (
    CallLLM,
    LLMRequest,
    LLMResponse,
)
from conversation_engine.infrastructure.llm.stub import call_llm_stub
from conversation_engine.infrastructure.llm.openai_adapter import (
    make_openai_llm,
    OpenAICallLLM,
)
from conversation_engine.infrastructure.llm.validator import (
    LLMValidator,
    LLMValidatorReport,
    QuizResult,
    quiz_report_summary,
)
from conversation_engine.models.validation_quiz import ValidationQuiz

__all__ = [
    "CallLLM",
    "LLMRequest",
    "LLMResponse",
    "call_llm_stub",
    "make_openai_llm",
    "OpenAICallLLM",
    "LLMValidator",
    "LLMValidatorReport",
    "ValidationQuiz",
    "QuizResult",
    "quiz_report_summary",
]
