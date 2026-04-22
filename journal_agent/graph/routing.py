import logging
from typing import Callable

# journal_agent/graph/routing.py
from collections.abc import Mapping
from typing import Any, Callable
from langgraph.graph import END


from journal_agent.model.session import Status

logger = logging.getLogger(__name__)


def _route_base(state: Mapping[str, Any], *, next_node: str, on_completion: str = END) -> str:
    if state.get("status") == Status.ERROR:
        logger.warning("Routing to END (id=%s, error=%s)",
                       state.get("session_id", "unknown"),
                       state.get("error_message"))
        return END
    if state.get("status") == Status.COMPLETED:
        return on_completion
    return next_node


def goto(node: str, on_completion: str = END) -> Callable[[Mapping[str, Any]], str]:
    def _goto(state: Mapping[str, Any]) -> str:
        return _route_base(state, next_node=node, on_completion=on_completion)

    return _goto