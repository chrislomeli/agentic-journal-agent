"""
FastAPI chat endpoint for the Journal Agent.
Drop this into your existing FastAPI app or run standalone.

Install: pip install fastapi uvicorn
Run:     uvicorn server:app --reload --port 8000
"""

import asyncio
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI()

# Allow Vite dev server to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    session_id: str | None = None


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Streams SSE events back to the React frontend.
    Each event is:   data: {"text": "<chunk>"}\n\n
    End signal is:   data: [DONE]\n\n
    """

    async def generate():
        # ------------------------------------------------------------------
        # Replace this block with your actual LangGraph invocation, e.g.:
        #
        #   async for event in your_graph.astream(
        #       {"messages": request.messages},
        #       config={"configurable": {"session_id": request.session_id}},
        #   ):
        #       # Extract the AI text chunk from the event
        #       chunk = extract_text(event)
        #       if chunk:
        #           yield f"data: {json.dumps({'text': chunk})}\n\n"
        #
        # ------------------------------------------------------------------

        # Placeholder: echo the last user message back word-by-word
        last_user_msg = next(
            (m.content for m in reversed(request.messages) if m.role == "user"), ""
        )
        words = f"You said: {last_user_msg}".split()
        for word in words:
            yield f"data: {json.dumps({'text': word + ' '})}\n\n"
            await asyncio.sleep(0.05)  # simulate streaming latency

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
