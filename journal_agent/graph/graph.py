

import logging
from typing import Literal, Callable

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from journal_agent.storage import SessionStore

logger = logging.getLogger(__name__)

# ── Graph builder ─────────────────────────────────────────────────────────────

from .state import JournalState, STATUS_ERROR, STATUS_COMPLETED


def _make_get_ai_response(
        llm: BaseChatModel) -> Callable[..., dict]:
    def get_ai_response(state: JournalState) -> dict:
        print(llm.get_name())
        return {}

    return get_ai_response


def _make_get_user_input(
        session_store: SessionStore )-> Callable[..., dict]:
    def get_user_input(state: JournalState) -> dict:
        print(session_store.get_last_session_id())
        return {}

    return get_user_input


def _make_save_turn(
        session_store: SessionStore )-> Callable[..., dict]:
    def save_turn(state: JournalState) -> dict:
        print(session_store.get_last_session_id())
        return {}

    return save_turn


def route_on_user_input(state: JournalState) -> Literal["get_ai_response", "save_turn", "__end__"]:
    if state["status"] == STATUS_ERROR:
        logger.warning(
            "Warning goes here"
        )
        return "__end__"
    elif state["status"] == STATUS_COMPLETED:
        return "save_turn"
    return "get_ai_response"

def route_on_ai_response(state: JournalState) -> Literal["save_turn", "__end__"]:
    if state.status == "error":
        logger.warning(
            "Warning goes here"
        )
        return "__end__"

    return "save_turn"

def route_on_save_turn(state: JournalState) -> Literal["get_user_input", "__end__"]:
    if state["status"] == STATUS_ERROR:
        logger.warning(
            "Warning goes here"
        )
        return "__end__"
    elif state["status"] == STATUS_COMPLETED:
        return "__end__"
    return "get_user_input"


def build_journal_graph(
        llm: BaseChatModel,
        session_store: SessionStore,
):
    """

    """
    # noinspection PyTypeChecker
    builder = StateGraph(JournalState) # no_qa

    builder.add_node("get_user_input", _make_get_user_input(session_store=session_store))
    builder.add_node("get_ai_response", _make_get_ai_response(llm=llm, session_store=session_store) )
    builder.add_node("save_turn", _make_save_turn(session_store=session_store))

    builder.add_edge(START, "get_user_input")
    builder.add_conditional_edges("get_user_input", route_on_user_input)
    builder.add_conditional_edges("get_ai_response", route_on_ai_response)
    builder.add_conditional_edges("save_turn", route_on_save_turn)
    compiled = builder.compile()
    return compiled
