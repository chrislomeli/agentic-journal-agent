# we are doing two things here - retrieving context and caching turns - but this is just to keep it out of core logic and will be refactored
from journal_agent.storage.storage import JsonStore


class ProfileStore:
    def __init__(self):
        self._session_store = JsonStore("transcripts")

    def retrieve_transcript(self, profile: UserProfile | None = None) -> list[BaseMessage] | None:
        _messages: list[BaseMessage] | None = None

        if (latest_session_id := self._session_store.get_last_session_id()) is not None:
            if (retrieved_exchanges := self._session_store.load_session(latest_session_id)) is not None:
                _messages = to_messages(retrieved_exchanges)
        return _messages

    def store_cache(self, session_id: str):
        self._session_store.save_session(session_id, self._exchanges)
        self._exchanges = []





