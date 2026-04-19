from abc import ABC, abstractmethod

from journal_agent.graph.state import JournalState


class PromptTemplateBuilder(ABC):

    @abstractmethod
    def build(self, state: JournalState) -> str:
        raise NotImplementedError
