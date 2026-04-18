"""
prompts — named prompt templates for each agent role.

Each prompt lives in its own module and exports a `TEMPLATE` string.
This package is a thin registry on top of those modules. Look up
prompts by key via `get_prompt(key)`; the enum `PromptKey` is the
source of truth for valid keys.
"""

from __future__ import annotations


from enum import Enum

from . import (
    conversation,
    decomposer,
    exchange_classifier,
    extractor,
    guidance,
    intent_classifier,
    socratic,
    thread_classifier,
)
from ._helpers import TAXONOMY, taxonomy_json

__all__ = ["get_prompt", "taxonomy_json", "TAXONOMY"]

from ...model.session import PromptKey

_REGISTRY: dict[str, str] = {
    PromptKey.INTENT_CLASSIFIER.value:   intent_classifier.TEMPLATE,
    PromptKey.CONVERSATION.value:        conversation.TEMPLATE,
    PromptKey.SOCRATIC.value:            socratic.TEMPLATE,
    PromptKey.GUIDANCE.value:            guidance.TEMPLATE,
    PromptKey.DECOMPOSER.value:          decomposer.TEMPLATE,
    PromptKey.THREAD_CLASSIFIER.value:   thread_classifier.TEMPLATE,
    PromptKey.EXCHANGE_CLASSIFIER.value: exchange_classifier.TEMPLATE,
    PromptKey.FRAGMENT_EXTRACTOR.value:  extractor.TEMPLATE,
}


def get_prompt(key: str | PromptKey) -> str:
    """Return the prompt template for *key*, or raise KeyError."""
    lookup = key.value if isinstance(key, PromptKey) else key
    try:
        return _REGISTRY[lookup]
    except KeyError:
        raise KeyError(
            f"Unknown prompt key {lookup!r}. "
            f"Available: {sorted(_REGISTRY)}"
        ) from None
