"""
call_llm_stub — Deterministic stub implementation of the CallLLM protocol.

Returns canned responses for testing.  No LLM dependency.
The stub inspects the request context to produce plausible responses
that exercise the same code paths as a real LLM.
"""

from __future__ import annotations

from conversation_engine.infrastructure.llm.protocols import LLMRequest, LLMResponse


def call_llm_stub(request: LLMRequest) -> LLMResponse:
    """
    Deterministic stub — returns canned responses based on request context.

    Matches the CallLLM Protocol signature:
        (request: LLMRequest) -> LLMResponse
    """
    # If the request contains findings in context, summarise them
    findings = request.context.get("findings", [])
    if findings:
        lines = [f"I found {len(findings)} issue(s) to discuss:"]
        for i, f in enumerate(findings, 1):
            msg = f.get("message", str(f)) if isinstance(f, dict) else str(f)
            lines.append(f"  {i}. {msg}")
        content = "\n".join(lines)
    else:
        content = "All validations passed. The architecture looks complete."

    return LLMResponse(
        content=content,
        model="stub",
        usage={"prompt_tokens": 0, "completion_tokens": 0},
        success=True,
    )
