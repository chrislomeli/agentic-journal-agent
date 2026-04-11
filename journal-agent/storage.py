import dataclasses
import json
from abc import ABC, abstractmethod
from pathlib import Path
import datetime
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from model import Turn, Role

class DataStore(ABC):
    @abstractmethod
    def save_session(self, session_id: str, turn: list[Turn]):
        pass

    @abstractmethod
    def _load_session(self, session_id: str) -> list[Turn] | None:
        pass


class SessionStore(DataStore):
    _storage: dict[str, list[Turn]] = dataclasses.field(default_factory=dict)

    def save_session(self, session_id: str, turn: list[Turn]):
        self._storage[session_id] = turn

    def _load_session(self, session_id: str) -> list[Turn] | None:
        return self._storage.get(session_id) or None

    def load_session_messages(self, session_id: str) ->  list[BaseMessage]:
        session_data = self._load_session(session_id)
        messages: list[BaseMessage] = []

        if not session_data:
            return messages

        for value in session_data:
            if value.role == Role.HUMAN:
                messages.append(HumanMessage(content=value.content))
            elif value.role == Role.AI:
                messages.append(AIMessage(content=value.content))

        return messages

    def load_session_turns(self, session_id: str) ->  list[Turn]:
        return self._storage.get(session_id) or []


class SessionStoreJSON(DataStore):

    _path: Path

    def __init__(self, path_name: str):
        if not path_name.endswith(".jsonl"):
            raise ValueError("Path must end with .jsonl")
        self._path = Path(path_name)


    def save_session(self, session_id: str, turn: list[Turn]):
        if self._path is None:
            raise ValueError("Path name is not set")

        if not self._path.parent.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)

        # Write line by line
        with self._path.open(mode="w", encoding="utf-8") as file:
            for line in turn:
                file.write(f"{line.model_dump_json()}\n")  # Manually add newline characters

    def _load_session(self, session_id: str) -> list[Turn] | None:
        data = []
        if self._path.exists():
            with self._path.open('r') as f:
                for line in f:
                    d = json.loads(line.strip())
                    t = Turn(exchange_id=d["exchange_id"], role=Role(d["role"]), content=d["content"], timestamp=datetime.datetime.fromisoformat(d["timestamp"]))
                    data.append(t)

        return data

    def load_session_messages(self, session_id: str) ->  list[BaseMessage]:
        session_data = self._load_session(session_id)
        messages: list[BaseMessage] = []

        if not session_data:
            return messages

        for value in session_data:
            if value.role == Role.HUMAN:
                messages.append(HumanMessage(content=value.content))
            elif value.role == Role.AI:
                messages.append(AIMessage(content=value.content))

        return messages

    def load_session_turns(self, session_id: str) ->  list[Turn]:
        return self._storage.get(session_id) or []