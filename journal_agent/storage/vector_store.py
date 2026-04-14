import json
from datetime import datetime

from journal_agent.model.session import Fragment, Tag


def fragment_to_chroma(f: Fragment) -> dict:
    return {
        "id": f.fragment_id,
        "document": f.content,
        "metadata": {
            "session_id": f.session_id,
            "exchange_ids": ",".join(f.exchange_ids),
            "tags": json.dumps([t.model_dump() for t in f.tags]),
            "timestamp": f.timestamp.isoformat(),
        },
    }

def fragment_from_chroma(row: dict) -> Fragment:
    meta = row["metadata"]
    return Fragment(
        fragment_id=row["id"],
        content=row["document"],
        session_id=meta["session_id"],
        exchange_ids=meta["exchange_ids"].split(",") if meta["exchange_ids"] else [],
        tags=[Tag(**t) for t in json.loads(meta["tags"])] if meta["tags"] else [],
        timestamp=datetime.fromisoformat(meta["timestamp"]),
    )