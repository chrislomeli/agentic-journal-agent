"""FastAPI application entry point.

Run with:
    uv run uvicorn journal_agent.api.main:app --reload

Then visit:
    http://localhost:8000/docs   ← interactive OpenAPI UI
"""
import logging

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from journal_agent.api.models import ChatRequest
from journal_agent.api.streaming import fake_stream

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Journal Agent API",
    description="Streaming chat interface for the journal agent.",
    version="0.1.0",
)


@app.post(
    "/chat/{session_id}",
    summary="Send a message and stream the response",
    response_description="Server-sent events stream of AI tokens",
)
async def chat(session_id: str, request: ChatRequest) -> StreamingResponse:
    """Send a message for a given session and receive a streaming SSE response.

    Each event in the stream has an `event` type and a `data` JSON payload:

    - **token** — one chunk of AI text: `{"text": "Hello "}`
    - **system** — feedback message: `{"text": "Saved 3 exchanges as 'topic'."}`
    - **done** — stream complete: `{"text": ""}`
    - **error** — something went wrong: `{"text": "error message"}`
    """
    logger.info("Chat request: session_id=%s message=%r", session_id, request.message)
    return StreamingResponse(
        fake_stream(request.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # prevents nginx from buffering the stream
        },
    )


@app.get("/health", summary="Health check")
async def health() -> dict:
    return {"status": "ok"}
