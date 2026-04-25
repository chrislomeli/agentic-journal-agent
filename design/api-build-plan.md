# Journal Agent API — Build Plan

## Context

This plan covers the build-out of the FastAPI layer for the journal agent. The
current state: a fully working CLI/LangGraph app with a thin FastAPI scaffold
(`api/main.py` with a fake SSE stream). The goal is a production-grade API that
the next phase (a UI) can sit on top of.

The architecture was designed under a "senior AI architect" lens — LangGraph is
the chosen framework, but used deliberately rather than reflexively. Concretely:

- Per-request graph invocation. **No `interrupt()` ever.**
- Checkpointer for in-flight state, repositories for the canonical archive.
- Conversation graph and end-of-session graph as two compiled graphs.
- Side effects (terminal I/O) pulled out of nodes; the runner does I/O.
- Streaming via LangGraph's `astream_events(version="v2")` — anticipating UI
  growth (status indicators, possible tool calling).

---

## Design Matrix — Active Decisions / Build Tasks

| # | Decision or task | Severity | Effort / Risk | Model | What it means concretely |
|---|---|---|---|---|---|
| **9a** | Decouple node I/O from the terminal | High | **M / M** | Sonnet (Opus reduces risk on async edge cases) | Refactor `get_ai_response` to yield tokens through an async generator. Update terminal client and API to consume it. Async streaming has subtle pitfalls (cancellation, backpressure, mid-stream errors). |
| **1** | EOS pipeline: collapse to one node, linear inside | Medium-High | **S / S** | Sonnet | Replace 7-node chain with single `end_of_session_node` calling each phase as plain async function. Structured logs at phase boundaries preserve observability. Conversation graph routes via one edge. |
| **4** | Adopt checkpointer for in-flight state; repositories canonical archive | Medium | **S / S** | Sonnet | `AsyncPostgresSaver` with `thread_id = session_id` (or `user_id:session_id` for multi-tenant). Live `JournalState` between requests. Repositories stay queryable archive. EOS node bridges checkpoint → repositories. Add cleanup job for closed sessions. |
| **3** | Keep classifier-router topology — deliberate, not default | Medium | **XS / XS now, M / M later** | N/A | No work today. Note as "kept pending tool-calling comparison." Latent risk: if a future eval shows tool-calling beats the classifier, refactor touches the conversation graph spine. |
| **5** | Streaming API: `astream_events(version="v2")` | Low | **S / S** | Sonnet | Anticipate growth (UI status indicators, possible tool calling). Consumer switches on event type; pass `on_chat_model_stream` as SSE token events; reserve other events for future UI features. |
| **7-design** | Make the system evaluable | (supports Critical hardening) | **S / XS** | Sonnet | Audit prompts and classifiers for structured outputs. Confirm prompt versioning. Confirm stable IDs. Patch gaps. Additive, low blast radius. |
| **8-design** | Make the system observable | (supports Critical hardening) | **M / XS** | Sonnet | Structured log fields, decision-point logs, operation-boundary entry/exit. Effort medium because of file count, not per-change difficulty. |

---

## Hardening Matrix — Sequenced After Core API Works

| # | Item | Severity | Trigger to start |
|---|---|---|---|
| **7-hardening** | Eval harness for classifier / decomposer / extractor / fragment_extractor | Critical | Before any prompt or model swap. Before scaling user count. |
| **8-hardening** | Cost/latency telemetry: per-role token counts, per-node latency, retry/failure metrics | Critical | Before the API serves real traffic. |
| **6** | Diversify observability: structured logs + metrics alongside LangSmith | Medium | Same time as #8-hardening — built on the same logging substrate. |

---

## Items Closed Out (no action)

- **#2** — `save_*` nodes: collapses into #1 automatically.
- **#9b** — DB writes inside nodes: idiomatic LangGraph, leave as is.
- **#10** — over-decomposition: judgment call, no specific node flagged.

---

## Suggested Sequencing

The matrix is ordered by severity, not build order. A reasonable build sequence:

1. **Cheap design wins first** to settle into patterns: #4 (checkpointer wiring),
   #1 (EOS collapse), #7-design (evaluability audit).
2. **The big design lift**: #9a (I/O decoupling) — most attention here.
3. **Streaming on top of the new node**: #5 (`astream_events` v2 consumer).
4. **Cross-cutting slog**: #8-design (observability) — benefits from a stable
   codebase to instrument.
5. **Hardening matrix** sequenced after the API is functioning end-to-end.

---

## Why These Decisions (Summary)

- **No interrupts**: couples graph execution to HTTP request lifecycle awkwardly.
  Per-request invocation gives clean HTTP semantics.
- **Checkpointer kept**: simpler than building incremental session persistence.
  No "two sources of truth" because checkpointer holds in-flight state and
  repositories hold the archive — different lifecycle stages.
- **EOS as one node**: the pipeline is linear ETL with no branching. A graph of
  7 nodes pretends it's a routing problem. One node honestly represents one
  unit of work.
- **`astream_events(v2)`**: project trajectory points to UI growth and likely
  richer event surfaces (status indicators, tool calling). The migration cost
  from `messages` mode to `v2` is real; better to start where you'll land.
- **Classifier-router kept for now**: defensible pattern with cost/latency
  advantages over tool-calling. Worth a future eval comparison; not a current
  rewrite.
