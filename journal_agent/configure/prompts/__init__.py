"""
prompts — named prompt templates for each agent role.

Each prompt lives in its own module. Most export a `TEMPLATE` string that is
fully resolved at import time (static prompts). A few export a
`PromptTemplate` with runtime placeholders (parametric prompts) — these are
formatted on each call with caller-supplied variables.

Look up prompts by key via `get_prompt(key, **vars)`. The enum `PromptKey`
is the source of truth for valid keys.
"""

# from _helpers import TAXONOMY, taxonomy_json

from journal_agent.configure.prompts import (
    conversation,
    decomposer,
    exchange_classifier,
    extractor,
    guidance,
    intent_classifier,
    profile_scanner,
    socratic,
    thread_classifier, verify_insights, label_clusters,
)

__all__ = ["get_prompt"]

from .base_prompt_template import PromptTemplateBuilder
from journal_agent.graph.state import JournalState
from journal_agent.model.session import PromptKey
from .conversation import ConversationProfileTemplate
from .socratic import SocraticProfileTemplate
from .guidance import GuidanceProfileTemplate
from .profile_scanner import UserProfileTemplate

_STATIC_REGISTRY: dict[str, str] = {
    PromptKey.INTENT_CLASSIFIER.value: intent_classifier.TEMPLATE,
    PromptKey.DECOMPOSER.value: decomposer.TEMPLATE,
    PromptKey.THREAD_CLASSIFIER.value: thread_classifier.TEMPLATE,
    PromptKey.EXCHANGE_CLASSIFIER.value: exchange_classifier.TEMPLATE,
    PromptKey.FRAGMENT_EXTRACTOR.value: extractor.TEMPLATE,
    PromptKey.VERIFY_INSIGHTS.value: verify_insights.TEMPLATE,
    PromptKey.LABEL_CLUSTERS.value: label_clusters.TEMPLATE,
}

_TEMPLATE_REGISTRY: dict[str, PromptTemplateBuilder] = {
    PromptKey.PROFILE_SCANNER.value: UserProfileTemplate(),
    PromptKey.CONVERSATION.value: ConversationProfileTemplate(),
    PromptKey.SOCRATIC.value: SocraticProfileTemplate(),
    PromptKey.GUIDANCE.value: GuidanceProfileTemplate(),
}


def get_prompt(key: str | PromptKey, state: JournalState | None = None) -> str:
    """Return the prompt for *key* as a fully-formatted string.

    Static prompts ignore ``prompt_vars``. Parametric prompts call
    ``PromptTemplate.format(**prompt_vars)`` and surface any missing-variable
    errors from LangChain.
    """
    lookup = key.value if isinstance(key, PromptKey) else key

    if lookup in _STATIC_REGISTRY:
        return _STATIC_REGISTRY[lookup]

    if state is None:
        raise ValueError("State is required for parametric prompts")

    if lookup in _TEMPLATE_REGISTRY:
        return _TEMPLATE_REGISTRY[lookup].build(state)

    # available = sorted(set(_STATIC_REGISTRY) | set(_TEMPLATE_REGISTRY))
    raise KeyError(
        f"Unknown prompt key {lookup!r}."
    )

