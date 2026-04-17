"""Tests for ContextBuilder."""

from datetime import datetime

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from journal_agent.configure.context_builder import (
    ContextBuildError,
    ContextBuilder,
    ContextTooLargeError,
    MissingStateError,
)
from journal_agent.model.session import Fragment, Tag


def make_fragment(content: str = "past thought", tags: tuple[str, ...] = ("general",)) -> Fragment:
    return Fragment(
        session_id="sess-1",
        content=content,
        exchange_ids=[],
        tags=[Tag(tag=t) for t in tags],
        timestamp=datetime.now(),
    )


def make_state(
    session_messages=None,
    seed_context=None,
    retrieved_history=None,
) -> dict:
    return {
        "session_id": "sess-1",
        "session_messages": session_messages or [],
        "seed_context": seed_context or [],
        "retrieved_history": retrieved_history or [],
        "transcript": [],
        "threads": [],
        "classified_threads": [],
        "fragments": [],
        "status": "processing",
        "error_message": None,
    }


class TestHappyPath:
    def test_builds_system_prefix_then_messages_in_order(self):
        builder = ContextBuilder()
        state = make_state(
            session_messages=[HumanMessage(content="hello")],
            seed_context=[HumanMessage(content="seed-msg")],
            retrieved_history=[make_fragment("past thought")],
        )

        messages = builder.get_context("conversation", state)

        assert isinstance(messages[0], SystemMessage)
        assert messages[1].content == "seed-msg"
        assert messages[2].content == "hello"

    def test_system_message_wraps_prompt_and_retrieved_context_in_xml(self):
        builder = ContextBuilder()
        state = make_state(retrieved_history=[make_fragment("remembered this")])

        messages = builder.get_context("conversation", state)
        sys_content = messages[0].content

        assert "<instructions>" in sys_content
        assert "</instructions>" in sys_content
        assert "<retrieved_context>" in sys_content
        assert "remembered this" in sys_content


class TestTokenBudget:
    def test_drops_retrieved_context_when_over_budget(self, monkeypatch):
        monkeypatch.setattr(
            "journal_agent.configure.context_builder.get_prompt",
            lambda key: "short prompt",
        )
        builder = ContextBuilder()
        builder.max_tokens = 100
        big_fragment = make_fragment(content="x" * 1000)
        state = make_state(
            session_messages=[HumanMessage(content="q")],
            retrieved_history=[big_fragment],
        )

        messages = builder.get_context("conversation", state)

        assert "<retrieved_context>" not in messages[0].content
        assert any(m.content == "q" for m in messages[1:])

    def test_trims_recent_messages_when_still_over_after_retrieved_dropped(self, monkeypatch):
        monkeypatch.setattr(
            "journal_agent.configure.context_builder.get_prompt",
            lambda key: "short prompt",
        )
        builder = ContextBuilder()
        builder.max_tokens = 50
        state = make_state(
            session_messages=[HumanMessage(content="keep-me")],
            seed_context=[HumanMessage(content="x" * 400)],
        )

        messages = builder.get_context("conversation", state)

        non_system = [m.content for m in messages[1:]]
        assert "keep-me" in non_system
        assert not any(c.startswith("x" * 100) for c in non_system)

    def test_raises_context_too_large_when_prompt_alone_blows_budget(self, monkeypatch):
        monkeypatch.setattr(
            "journal_agent.configure.context_builder.get_prompt",
            lambda key: "x" * 10_000,
        )
        builder = ContextBuilder()
        builder.max_tokens = 100
        state = make_state()

        with pytest.raises(ContextTooLargeError) as exc_info:
            builder.get_context("conversation", state)

        err = exc_info.value
        assert err.tokens > err.budget
        assert isinstance(err, ContextBuildError)


class TestErrors:
    def test_missing_prompt_key_raises_missing_state_error(self):
        builder = ContextBuilder()
        state = make_state()

        with pytest.raises(MissingStateError) as exc_info:
            builder.get_context("no_such_prompt_key_ever", state)

        assert "prompt:" in str(exc_info.value)
        assert isinstance(exc_info.value, ContextBuildError)

    def test_missing_state_key_raises_missing_state_error(self):
        builder = ContextBuilder()
        state = {"session_id": "sess-1"}

        with pytest.raises(MissingStateError):
            builder.get_context("conversation", state)


class TestTokenCounting:
    def test_estimate_string_tokens_is_chars_div_4(self):
        builder = ContextBuilder()
        assert builder.count_string_tokens("a" * 40) == 10

    def test_estimate_message_tokens_includes_per_message_overhead(self):
        builder = ContextBuilder()
        msg = HumanMessage(content="a" * 40)

        assert builder.count_message_tokens([msg]) == 15
        assert builder.count_message_tokens([msg, msg]) == 30


class TestStateIsolation:
    def test_pruning_does_not_mutate_caller_state_lists(self, monkeypatch):
        monkeypatch.setattr(
            "journal_agent.configure.context_builder.get_prompt",
            lambda key: "short",
        )
        builder = ContextBuilder()
        builder.max_tokens = 50
        state = make_state(
            session_messages=[HumanMessage(content="x" * 200)],
            seed_context=[HumanMessage(content="y" * 200)],
        )

        try:
            builder.get_context("conversation", state)
        except ContextTooLargeError:
            pass

        assert len(state["session_messages"]) == 1
        assert len(state["seed_context"]) == 1
