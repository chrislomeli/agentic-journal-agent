import logging

from langgraph.graph import END, START, StateGraph

from journal_agent.comms.llm_registry import LLMRegistry
from journal_agent.graph.nodes.insight_nodes import make_cluster_fragments, make_label_clusters, make_verify_citations, \
    make_collect_window, make_format_result
from journal_agent.graph.nodes.stores import (
    make_save_insights,
)
from journal_agent.graph.routing import goto

logger = logging.getLogger(__name__)

from journal_agent.graph.state import ReflectionState
from journal_agent.stores import PgFragmentRepository, InsightsRepository


def build_reflection_graph(
        registry: LLMRegistry,
        fragment_store: PgFragmentRepository,
        insights_repo: InsightsRepository,
):
    """todo add comments

    """

    reflection_llm = registry.get("classifier")


    # noinspection PyTypeChecker
    builder = StateGraph(ReflectionState)  # no_qa

    # Reflection pipeline nodes
    builder.add_node("collect_window", make_collect_window(fragment_store=fragment_store))
    builder.add_node("cluster_fragments", make_cluster_fragments())
    builder.add_node("label_clusters", make_label_clusters(llm=reflection_llm))
    builder.add_node("verify_citations", make_verify_citations(llm=reflection_llm, max_concurrency=3))
    builder.add_node("persist_insights", make_save_insights(insights_repo=insights_repo))
    builder.add_node("format_result", make_format_result())

    # Wiring
    builder.add_edge(START, "collect_window")
    builder.add_conditional_edges("collect_window", goto("cluster_fragments"))
    builder.add_conditional_edges("cluster_fragments", goto("label_clusters"))
    builder.add_conditional_edges("label_clusters", goto("verify_citations"))
    builder.add_conditional_edges("verify_citations", goto("persist_insights"))
    builder.add_conditional_edges("persist_insights", goto("format_result"))
    builder.add_edge("format_result", END)

    compiled = builder.compile()
    return compiled
