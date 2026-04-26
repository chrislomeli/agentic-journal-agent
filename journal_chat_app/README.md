# Journal Agent Chat UI

React + TypeScript + Vite chat frontend for your LangGraph/FastAPI backend.

## Structure

```
chat-app/
├── src/
│   ├── hooks/
│   │   └── useChat.ts          # SSE streaming hook — wire your custom logic here
│   ├── components/
│   │   ├── Chat.tsx            # Main chat layout
│   │   ├── MessageBubble.tsx   # Message rendering
│   │   └── ChatInput.tsx       # Input + send/stop buttons
│   ├── types/index.ts          # Shared types
│   ├── App.tsx
│   ├── main.tsx
│   └── index.css
├── server.py                   # FastAPI example backend (replace with yours)
├── vite.config.ts              # Proxies /api → localhost:8000
├── package.json
└── tsconfig.json
```

## Quick Start

### 1. Install and run the React app

```bash
npm install
npm run dev
# → http://localhost:5173
```

### 2. Run the FastAPI backend

```bash
pip install fastapi uvicorn
uvicorn server:app --reload --port 8000
```

### 3. Wire up your LangGraph app

In `server.py`, replace the placeholder generator with your actual LangGraph stream:

```python
async for event in your_graph.astream({"messages": request.messages}):
    chunk = extract_text(event)
    if chunk:
        yield f"data: {json.dumps({'text': chunk})}\n\n"
```

## Adding Custom SSE Event Types

In `src/hooks/useChat.ts`, find the SSE parsing loop and add handlers:

```typescript
// Example: handle a custom node_update event from LangGraph
if (line.startsWith('event: node_update')) {
  // next line will be the data
}
```

On the FastAPI side, emit typed events:

```python
yield f"event: node_update\ndata: {json.dumps(state)}\n\n"
yield f"event: message\ndata: {json.dumps({'text': chunk})}\n\n"
```

## Key Files to Customize

| File | What to change |
|------|---------------|
| `src/hooks/useChat.ts` | SSE parsing, custom event hooks, session management |
| `server.py` | Swap placeholder with your LangGraph `astream` call |
| `src/components/Chat.tsx` | Layout, header, empty state |
| `src/components/MessageBubble.tsx` | Message styling |
| `vite.config.ts` | Change proxy target if FastAPI runs on a different port |
