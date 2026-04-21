# Phase 10: Reflective Memory Layer — Design & Build Plan

## Context

By end of Phase 9, the journal agent has:
- Per-turn conversation with retrieval and intent-shaped context
- Fragment extraction (episodic memory)
- User profile (stable preferences/traits)

Phase 10 adds **reflective memory** — a layer that aggregates across fragments to surface patterns, themes, and cross-session observations the user hasn't named themselves.

Think of it as the layer that lets the agent say *"you keep coming back to the tension between X and Y"* — observations that only exist because the system aggregated across many conversations.

---

## Product thesis: pull, not push

The system does **not** proactively surface reflections. The user invokes reflection on demand via CLI commands:

- `/reflect` — surface current themes across the fragment corpus
- `/recall <topic>` — retrieve and cite fragments matching a topic
- `/recap` — summarize the current session

**Why pull over push:** Push-based surfacing requires a "novelty filter" — the system must know what's worth interrupting the user with. That filter is an open research problem. Pull sidesteps it entirely: the user signals when a question is live, and a 70%-good answer to a live question feels delightful. The same answer pushed at the wrong moment feels noisy.

Push remains gated behind evidence we don't yet have — namely, engagement signal on pull-surfaced insights. Post-POC feature, not POC scope.

---

## Feature scope

### Must-have (core differentiation)

1. **Cross-session pattern surfacing** — "you keep coming back to X"
2. **Theme synthesis / reflection engine** — hierarchical map-reduce over fragments
3. **Provenance on every insight** — every observation traces back to source fragments

### Should-have (trust and polish)

4. **Memory inspection + editability** — see what the system knows, correct it
5. **Verifier** — every insight's citations are re-checked before surfacing
6. **Importance-weighted consolidation** — low-value fragments decay, high-value promote into reflections

### Deferred / explicitly NOT in POC

- **Push-based surfacing** — gated on novelty signal we don't have
- **Drift detection / contradiction detection** — requires temporal KG (Zep/Graphiti); deferred
- **Entity/relationship graphs** — wrong market for introspective journaling
- **Affect/emotion tracking** — scope creep into companion-app territory
- **Open-loop / goal tracking** — orthogonal machinery, separate feature
- **Multi-user / auth** — journal is single-tenant by design
- **Web UI** — CLI is sufficient for POC

---

## Architectural components

Nine product features collapse to five architectural components. POC builds two of them:

| Component | Powers features | In POC? |
|---|---|---|
| **A. Hierarchical reflection (map-reduce + clustering)** | Pattern surfacing, theme synthesis, importance consolidation | ✓ |
| **B. Temporal entity store** | Drift, contradictions | — (deferred) |
| **C. Provenance-bearing records + cascading invalidation** | Inspection/editability, verifier | ✓ |
| **D. Schema-extracted state machines** | Open-loop tracking | — (deferred) |
| **E. Per-turn classification + rolling signals** | Affect awareness | — (deferred) |

**POC scope: A + C only.**

---

## System architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                         journal_agent                              │
│                                                                    │
│   ┌─────────────────────┐          ┌──────────────────────┐        │
│   │   Chat Graph        │          │  Reflection Graph    │        │
│   │ (per-turn, online)  │          │ (on-demand, batch)   │        │
│   │                     │          │                      │        │
│   │ input→intent→       │          │ collect→cluster→     │        │
│   │ retrieve→context→   │          │ score→label→         │        │
│   │ respond→extract     │          │ verify→persist       │        │
│   └─────────┬───────────┘          └──────────┬───────────┘        │
│             │                                 │                    │
│             ▼                                 ▼                    │
│   ┌─────────────────────────────────────────────────────┐          │
│   │              Memory Facade (async Protocols)        │          │
│   │                                                     │          │
│   │   FragmentStore   InsightStore   ProfileStore       │          │
│   │   (episodic)      (derived)      (stable)           │          │
│   │      │                │               │             │          │
│   │   Postgres +      Postgres +      profile.json      │          │
│   │   pgvector        relational                        │          │
│   └─────────────────────────────────────────────────────┘          │
└────────────────────────────────────────────────────────────────────┘
```

**Three stores, three access patterns, appropriately different backends.**

The load-bearing design decision: rejecting "everything in one vector DB." Fragment access is similarity search. Insight access is relational + cascade. Profile access is key-value. Forcing all three into one backend teaches the wrong architectural lesson.

---

## Data model

```python
class Fragment(BaseModel):
    id: FragmentId
    content: str
    tags: list[str]
    summary: str
    created_at: datetime
    source_turn_id: TurnId
    session_id: SessionId
    importance: float               # 0–1, LLM-rated at write-time
    embedding: Vector               # for similarity search

class Insight(BaseModel):
    id: InsightId
    label: str                      # LLM-generated theme
    body: str                       # the actual observation
    source_fragment_ids: list[FragmentId]   # REQUIRED — non-optional
    confidence: float
    generated_at: datetime
    verifier_status: Literal["unverified", "verified", "failed"]

class UserProfile(BaseModel):
    # carries forward from Phase 9
    preferences: dict
    themes: list[str]
    traits: list[str]
```

**Discipline:** `source_fragment_ids` is **non-optional** on `Insight`. No derived record without traceable evidence. This is the schema-level enforcement of the provenance principle.

---

## LangGraph topology — two graphs, shared stores

**Chat graph (per-turn, online):** unchanged from Phase 9 except it writes into the new `FragmentStore` via the Protocol (not directly into Chroma).

**Reflection graph (on-demand, batch):**
```
START → collect_window → cluster_fragments → score_clusters →
        label_clusters → verify_citations → persist_insights → summarize_result → END
```

Both graphs compile from the same `graph_utils` factory. Reflection graph runs via:
- CLI command (`/reflect`) — primary trigger
- Session close — optional, configurable

**Not a node inside the chat graph.** Different lifecycle (batch vs online), different failure semantics (fan-out isolated vs per-turn fail-fast), different observability needs.

---

## Node reference

### Chat graph — Conversation loop (runs every turn)

| Node | Factory function | Reads from state | Writes to state | What it does |
|---|---|---|---|---|
| `get_user_input` | `make_get_user_input` | `session_id` | `session_messages`, `status` | Reads console input. `/quit` sets `status=COMPLETED` and exits the loop into the end-of-session pipeline. Otherwise records the human turn and appends a `HumanMessage`. |
| `intent_classifier` | `make_intent_classifier` | `session_messages`, `user_profile` | `context_specification` | Sends recent messages to LLM → produces a `ScoreCard`. Resolves to `ContextSpecification`: which prompt to use, how many fragments to retrieve (`top_k`), how tight the similarity filter (`distance`). Also flags `user_profile.is_current = False` if `personalization_score >= 0.5`, which triggers `profile_scanner` on the next routing decision. |
| `profile_scanner` | `make_profile_scanner` | `session_messages`, `user_profile` | `user_profile` | **Conditional** — only runs when `user_profile.is_current == False`. Sends recent messages to LLM to refresh UserProfile with new preferences/traits. Saves to ProfileStore. Skips if profile is already current. |
| `retrieve_history` | `make_retrieve_history` | `session_messages`, `context_specification` | `retrieved_history` | **Conditional** — only runs when `top_k_retrieved_history > 0`. Queries FragmentStore with the latest human message + intent tags as the query vector. Returns top-k most semantically relevant fragments. |
| `get_ai_response` | `make_get_ai_response` | `session_messages`, `context_specification`, `retrieved_history`, `user_profile` | `session_messages`, `transcript`, `status` | Assembles the full prompt via `ContextBuilder` (system + profile + retrieved fragments + recent turns + current input). Calls conversation LLM. Records the exchange. Prints the response. Loops back to `get_user_input`. |

**Routing in the conversation loop:**
```
get_user_input
  → /quit              → save_transcript (exits loop)
  → else               → intent_classifier

intent_classifier
  → profile stale      → profile_scanner
  → top_k > 0          → retrieve_history
  → else               → get_ai_response

profile_scanner
  → top_k > 0          → retrieve_history
  → else               → get_ai_response

retrieve_history       → get_ai_response
get_ai_response        → get_user_input  (loops)
```

---

### Chat graph — End-of-session pipeline (runs once after `/quit`)

**Important:** Fragment extraction happens at end-of-session, not per-turn. Fragments from the current session are NOT available for retrieval until after the user quits.

| Node | Factory function | Reads from state | Writes to state | What it does |
|---|---|---|---|---|
| `save_transcript` | `make_save_transcript` | `transcript` | — | Persists all session exchanges to storage. Gateway into the end-of-session pipeline. |
| `exchange_decomposer` | `make_exchange_decomposer` | `transcript` | `threads` | Sends full transcript to LLM. LLM groups exchanges into topical `ThreadSegment`s — clusters of related exchanges by topic. |
| `save_threads` | `make_save_threads` | `threads` | — | Persists raw (unclassified) thread segments. |
| `thread_classifier` | `make_thread_classifier` | `threads`, `transcript` | `classified_threads` | Async fan-out: one LLM call per thread (bounded by semaphore). Assigns taxonomy tags to each thread. |
| `save_classified_threads` | `make_save_classified_threads` | `classified_threads` | — | Persists tagged thread segments. |
| `thread_fragment_extractor` | `make_thread_fragment_extractor` | `classified_threads`, `transcript` | `fragments` | Async fan-out: one LLM call per thread. Distills each thread into 0-N standalone `Fragment` records (atomic ideas, tags, provenance back to exchange IDs). |
| `save_fragments` | `make_save_fragments` | `fragments` | — | Writes all new Fragments to FragmentStore. Now available for retrieval in future sessions. |

**Pipeline is linear — no conditional routing:**
```
save_transcript → exchange_decomposer → save_threads
  → thread_classifier → save_classified_threads
  → thread_fragment_extractor → save_fragments → END
```

---

### Reflection graph — On-demand (triggered by `/reflect`)

All nodes are new in Phase 10. The fragment pool loaded by `collect_window` is shared by both the labeler and verifier — one fetch, two consumers.

| Node | Reads from state | Writes to state | What it does |
|---|---|---|---|
| `collect_window` | `window_start`, `window_end` | `fragments` | `FragmentStore.all_since(window_start)` — loads all fragments in the time window. This is the shared pool for all downstream nodes. |
| `cluster_fragments` | `fragments` | `clusters` | HDBSCAN density clustering on fragment embeddings. Produces variable-k groups of semantically related fragments. No fixed cluster count — it's data-driven. |
| `score_clusters` | `clusters` | `scored_clusters` | Scores each cluster: `size × avg_importance × recency_spread`. Drops clusters below threshold. The importance filter against the trivia problem. |
| `label_clusters` | `scored_clusters`, `fragments` | `insights` | For each surviving cluster: calls LLM with that cluster's fragments → produces `Insight(label, body, source_fragment_ids)`. Insights are unverified at this point. |
| `verify_citations` | `insights`, `fragments` | `verified_insights` | For each insight: looks up cited fragment IDs from the in-memory pool (same pool as `collect_window` loaded). Calls verifier LLM: "does this evidence support this claim?" Marks `verifier_status = verified or failed`. Only passes verified insights forward. |
| `persist_insights` | `verified_insights` | — | Writes each verified Insight to InsightStore. Now available for the chat graph's `retrieve_history` node. |
| `summarize_result` | `verified_insights` | `summary` | Formats a human-readable summary for CLI output. |

**Graph is linear:**
```
START → collect_window → cluster_fragments → score_clusters
      → label_clusters → verify_citations → persist_insights
      → summarize_result → END
```

---

## Scenarios

Concrete user actions traced through every node that fires.

---

### Scenario A: User asks a question (normal chat turn)

**User types:** `"I've been wrestling with whether to take this new job — what do you think?"`

```
get_user_input
  ← reads user message
  → session_messages += [HumanMessage("I've been wrestling...")]
  → status = PROCESSING

intent_classifier
  ← session_messages (last 5), user_profile
  → LLM produces ScoreCard: evaluation-leaning, personalization_score = 0.3
  → resolves to ContextSpecification: {top_k: 5, distance: 1.1, prompt_key: EVALUATION}
  → personalization_score < 0.5 → profile stays current

  router: profile current AND top_k=5 > 0 → retrieve_history

retrieve_history
  ← query = "job offer decision" + intent tags
  → FragmentStore.search(query, top_k=5, distance=1.1)
  → returns 5 fragments from prior sessions about career thinking
  → retrieved_history = [f1, f2, f3, f4, f5]

get_ai_response
  ← session_messages, retrieved_history, user_profile
  → ContextBuilder assembles:
      [system prompt + profile traits]
      [5 retrieved fragments re: career]
      [last 2 session turns]
      [current user message]
  → conversation LLM generates response
  → response printed to console
  → exchange appended to transcript

  router: status = PROCESSING → back to get_user_input
```

**What the user experiences:** AI responds with awareness of past career thoughts. Nothing surfaced explicitly — retrieved context is ambient.

---

### Scenario B: User types `/quit`

```
get_user_input
  ← reads "/quit"
  → status = COMPLETED

  router: COMPLETED → save_transcript

save_transcript
  → full session transcript persisted

exchange_decomposer
  ← transcript (all exchanges)
  → LLM identifies 3 topical threads: "job decision", "project X", "weekend plans"
  → threads = [thread_1, thread_2, thread_3]

save_threads → persists raw threads

thread_classifier  [async fan-out: 3 concurrent LLM calls]
  → thread_1 tagged: [career, decision, evaluation]
  → thread_2 tagged: [project, planning, technical]
  → thread_3 tagged: [personal, lifestyle]
  → classified_threads = [tagged_1, tagged_2, tagged_3]

save_classified_threads → persists tagged threads

thread_fragment_extractor  [async fan-out: 3 concurrent LLM calls]
  → thread_1 → 2 fragments (career decision thoughts)
  → thread_2 → 3 fragments (project planning ideas)
  → thread_3 → 1 fragment  (weekend preference note)
  → fragments = [f1, f2, f3, f4, f5, f6]

save_fragments
  → 6 new Fragments written to FragmentStore
  → available for retrieval in future sessions

→ END
```

**What the user experiences:** session ends cleanly. Pipeline runs silently. 6 fragments now in the store.

---

### Scenario C: User types `/reflect`

```
CLI: builds ReflectionState(window_start=now-30d, window_end=now)

collect_window
  → FragmentStore.all_since(now - 30d)
  → loads 180 fragments into state["fragments"]  ← shared pool

cluster_fragments
  ← 180 fragment embeddings
  → HDBSCAN finds 9 clusters

score_clusters
  ← 9 clusters
  → 4 survive threshold (large, high importance, temporally spread)
  → 5 dropped (too small or low importance)

label_clusters
  ← 4 surviving clusters + state["fragments"] pool
  → LLM call per cluster:
      cluster 1 → Insight("job decision tension", "you keep circling back...", [f12, f45, f78])
      cluster 2 → Insight("architecture vs delivery", "...", [f23, f56])
      cluster 3 → Insight("learning approach", "...", [f34, f67, f89])
      cluster 4 → Insight("work-life balance", "...", [f90, f91])
  → state["insights"] = 4 unverified insights

verify_citations
  ← 4 insights + state["fragments"] pool
  → per insight, look up cited IDs from pool, call verifier LLM:
      insight 1: supported=true  ✓
      insight 2: supported=true  ✓
      insight 3: supported=false  ✗ (over-generalized — 2 fragments cited, claim too broad)
      insight 4: supported=false  ✗ (topic drift — fragments about scheduling, not balance)
  → state["verified_insights"] = 2 insights

persist_insights
  → 2 verified Insights written to InsightStore
  → now retrievable by chat graph

summarize_result
  → formats top 2 insights for output

CLI prints:
  "2 themes found in your last 30 days:

   1. job decision tension
      You keep circling back to the growth-vs-stability tension...
      Sources: [2026-04-01] f12, [2026-04-10] f45, [2026-04-15] f78

   2. architecture vs delivery
      ..."
```

---

### Scenario D: User types `/recall career`

**No graph runs.**

```
CLI: FragmentStore.search("career", k=10)
  → returns 10 most semantically similar fragments

CLI prints:
  "[2026-04-01] 'I've been wrestling with whether the new role...' (session: morning-session)
   [2026-03-28] 'The stability vs growth question came up again...'
   ..."
```

Plain retrieval. No clustering, no labeling, no insights generated.

---

### Scenario E: User types `/insights`

**No graph runs.**

```
CLI: InsightStore.recent(since=now-30d, limit=10)
  → returns stored verified insights

CLI prints:
  "[generated: 2026-04-18] job decision tension
     You keep circling back to the growth vs stability tension...
     Sources: f12, f45, f78

   [generated: 2026-04-10] architecture vs delivery
     ..."
```

Reads what was stored by a prior `/reflect` run. Nothing is generated here.

---

### The three modes, summarised

| User action | Graph that runs | Stores touched | What gets created |
|---|---|---|---|
| Chat message | Chat graph (conversation loop) | Read: FragmentStore, ProfileStore | Exchange appended to session |
| `/quit` | Chat graph (end-of-session pipeline) | Write: FragmentStore | Fragments extracted from session |
| `/reflect` | Reflection graph | Read: FragmentStore; Write: InsightStore | Verified insights |
| `/recall <topic>` | None | Read: FragmentStore | Nothing new |
| `/insights` | None | Read: InsightStore | Nothing new |

---

## Storage decisions

**Postgres + pgvector** for FragmentStore and InsightStore.

- Docker image: `pgvector/pgvector:pg16` — Postgres 16 with extension preinstalled
- Setup: `docker compose up -d` (~15 seconds)
- Enables: real foreign keys, transactional cascade invalidation, SQL joins across stores

**JSON file** for ProfileStore.

- One mutable record; Postgres would be overkill
- `data/profile.json` — continues from Phase 9, wrapped behind Protocol

**Why this combination:**

Three stores, three access patterns:
- Fragment: semantic similarity + metadata filter → vector + relational
- Insight: time-range, FK lookup, status filter → relational (+ optional text search)
- Profile: single-record KV → file

Putting Fragment and Insight in the same Postgres lets you write `DELETE fragments WHERE … CASCADE` and have dependent insights invalidate transactionally. You cannot do this cleanly with Chroma + SQLite.

### Docker Compose

Add to repo root as `docker-compose.yml`:

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_PASSWORD: journal
      POSTGRES_USER: journal
      POSTGRES_DB: journal
    ports:
      - "5432:5432"
    volumes:
      - journal_pg:/var/lib/postgresql/data
volumes:
  journal_pg:
```

---

## Risks to watch for

Ordered by severity. These inform the build order and the testing discipline.

1. **The "so what" problem** — reflections are technically correct but surprise no one. Mitigation: prompt labeler for concrete language with verbatim snippets; avoid over-abstraction. Accept that solving this fully is an open problem.
2. **Hallucinated citations** — LLM invents source IDs that don't support the claim. Mitigation: **verifier node is mandatory**, not optional. Sub-phase 10.4.
3. **Cluster quality** — embedding clustering is not topic modeling. Mitigation: density-based (HDBSCAN) rather than fixed-k; tune threshold empirically.
4. **Over-abstraction** — "user values authenticity" when the threads were specific. Mitigation: prompt discipline + include verbatim fragment snippets in labels.
5. **Trivia problem** — noisy recurring themes crowd out substance. Mitigation: importance-weighted clustering via score_clusters node.
6. **Recency blindness** — map-reduce is time-agnostic. Mitigation: `collect_window` takes a time range; reflection can run at multiple windows.
7. **Cascade complexity** — invalidation logic gets hairy fast. Mitigation: mark-stale + lazy recompute, not eager recompute.
8. **Scale degradation** — works at 50 fragments, murky at 5000. Mitigation: out of scope for POC; note as known limit.
9. **Evaluation opacity** — no ground truth. Mitigation: golden-set regression harness (Sub-phase 10.4).

---

## Build plan — paint by numbers

Five sub-phases. Each is runnable and reviewable in isolation. Same rhythm as Phases 1–9: small steps, visible progress, review checkpoint.

---

### Sub-phase 10.1: Store refactor

**Goal:** Split current persistence into three Protocol-backed stores. Behavior unchanged — this is scaffolding.

**Files to create:**
- `journal_agent/stores/__init__.py`
- `journal_agent/stores/base.py` — the Protocols
- `journal_agent/stores/fragment_store.py` — Chroma adapter (keep for now)
- `journal_agent/stores/fragment_store_pgvector.py` — pgvector adapter (new, stub ok for this sub-phase)
- `journal_agent/stores/insight_store.py` — Postgres adapter
- `journal_agent/stores/profile_store.py` — JSON file adapter (wrap existing)

**Protocols to define:**

```python
# stores/base.py

class FragmentStore(Protocol):
    async def write(self, frag: Fragment) -> FragmentId: ...
    async def search(self, q: str, k: int, filters: Filters | None = None) -> list[Fragment]: ...
    async def by_ids(self, ids: list[FragmentId]) -> list[Fragment]: ...
    async def all_since(self, since: datetime) -> list[Fragment]: ...

class InsightStore(Protocol):
    async def write(self, ins: Insight) -> InsightId: ...
    async def recent(self, since: datetime, limit: int = 10) -> list[Insight]: ...
    async def invalidate_by_source(self, frag_ids: list[FragmentId]) -> int: ...
    async def by_label(self, label: str) -> list[Insight]: ...

class ProfileStore(Protocol):
    async def get(self) -> UserProfile: ...
    async def update(self, patch: ProfilePatch, source_ids: list[FragmentId]) -> None: ...
```

**Canonical pattern:** Protocol + Adapter. Graph code imports only the Protocol type. Concrete adapter injected at startup (composition root in `main.py`).

**Test criterion:**
- All existing Phase 9 tests pass unchanged
- Graph code has zero imports of `chromadb` or raw `json` — only `FragmentStore`/`InsightStore`/`ProfileStore`

---

### Sub-phase 10.2: Reflection graph skeleton

**Goal:** A runnable reflection graph with empty-but-correctly-typed nodes. End-to-end plumbing before logic.

**Files to create:**
- `journal_agent/reflection/__init__.py`
- `journal_agent/reflection/state.py` — `ReflectionState(TypedDict)`
- `journal_agent/reflection/nodes.py` — node stubs
- `journal_agent/reflection/builder.py` — wire the graph
- `journal_agent/reflection/runner.py` — convenience entrypoint

**State shape:**

```python
class ReflectionState(TypedDict):
    window_start: datetime
    window_end: datetime
    fragments: list[Fragment]            # populated by collect_window
    clusters: list[Cluster]              # populated by cluster_fragments
    scored_clusters: list[ScoredCluster] # populated by score_clusters
    insights: list[Insight]              # populated by label_clusters
    verified_insights: list[Insight]     # populated by verify_citations
    summary: str                         # populated by summarize_result
    error: str | None
```

**Node stubs:** each returns a placeholder value. The graph compiles and runs end-to-end, just produces nothing meaningful yet.

**Test criterion:** `python -m journal_agent.reflection.runner --since "2026-01-01"` runs cleanly and returns an empty result without crashing.

---

### Sub-phase 10.3: Clustering + labeling node

**Goal:** Make the graph actually produce meaningful insights from fragments.

**Implement one node at a time:**

1. **`collect_window`** — `FragmentStore.all_since(window_start)`. Straightforward.

2. **`cluster_fragments`** — vector clustering over fragment embeddings. Use `sklearn.cluster.HDBSCAN` (no fixed k required). Output: `list[Cluster]` where each cluster has `fragment_ids` and a centroid.
   - **Canonical pattern:** density-based clustering for unknown cluster count.

3. **`score_clusters`** — for each cluster, compute a score from: cluster size, average fragment importance, recency spread. Drop clusters below threshold.
   - **Canonical pattern:** this is your importance filter against the trivia problem. Non-optional.

4. **`label_clusters`** — for each surviving cluster, call LLM with cluster fragments and prompt it to produce `(label, body, source_fragment_ids)`. Use structured output (Pydantic `Insight` schema).
   - **Canonical pattern:** map-reduce with LLM labels. For POC, sequential loop is fine. Phase 11 converts to `asyncio.gather`.

**Prompt discipline for the labeler:**
- Demand concrete language with verbatim phrases from fragments
- Demand `source_fragment_ids` populated with actual fragments shown
- Explicitly reject over-abstraction ("say 'user is weighing X vs Y', not 'user values decisiveness'")

**Test criterion:** run `/reflect` on a seeded corpus of ~30 fragments spanning 3 obvious topics. Verify at least 2 of 3 topics are correctly labeled with fragment citations.

---

### Sub-phase 10.4: Verifier + golden set

**Goal:** Close the hallucinated-citation hole and build an evaluation harness.

**Implement:**

1. **`verify_citations` node:**

```python
async def verify_citations(state: ReflectionState, llm: LLMClient, store: FragmentStore):
    verified = []
    for ins in state["insights"]:
        sources = await store.by_ids(ins.source_fragment_ids)
        verdict = await llm.ainvoke(
            VERIFIER_PROMPT.format(claim=ins.body, evidence=sources),
            response_model=VerifierVerdict,
        )
        ins.verifier_status = "verified" if verdict.supported else "failed"
        if verdict.supported:
            verified.append(ins)
    return {"verified_insights": verified}
```

2. **Golden set:** hand-curate 20 fragments covering 3–4 obvious themes. Commit to `journal_agent/tests/fixtures/reflection_golden.json`.

3. **Regression test:** runs reflection graph on the golden fragments, asserts expected themes are surfaced and citations point back correctly.

**Test criterion:** CI passes the golden-set regression. Manually introduce a prompt regression and verify the test catches it.

---

### Sub-phase 10.5: Pull CLI surface

**Goal:** User-facing commands that invoke the reflection machinery.

**Implement in `journal_agent/main.py`:**

- `/reflect [--since N]` — run reflection graph, print top 3 insights with citations
- `/recall <topic>` — search FragmentStore, print matching fragments with session context
- `/recap` — run reflection on current session only
- `/insights` — list recent stored insights with source links

**Pattern:** CLI commands are thin wrappers that build initial `ReflectionState` (or call `FragmentStore.search` for `/recall`), invoke the appropriate graph, format output.

**Test criterion:**
- Seeded journal with 3 sessions across 3 topics
- `/reflect` surfaces coherent themes with citations
- `/recall "<specific phrase>"` returns relevant fragments
- Edit a fragment; re-run `/reflect`; verify affected insights marked stale (cascade invalidation works end-to-end)

---

## Interview-ready answers

Reviewer questions you should be able to answer in one sentence each:

- **"Why three stores?"** → Different access patterns: similarity vs relational vs key-value.
- **"Why Postgres + pgvector for two of them?"** → Foreign keys and transactional cascade across fragment→insight provenance.
- **"How do you handle hallucinated citations?"** → Verifier node re-checks every insight against source fragments before surfacing.
- **"What happens when a user edits a fragment?"** → `InsightStore.invalidate_by_source` marks dependent insights stale; next `/reflect` recomputes lazily.
- **"Why isn't reflection a node in the chat graph?"** → Different lifecycle (batch vs online), different failure semantics, user-triggered not per-turn.
- **"How do you know reflections are good?"** → Golden-set regression in CI + user click-through signal on pull surface (future).
- **"Why pull over push?"** → Push requires a novelty filter, which is unsolved. Pull lets the user signal when a question is live.

---

## What you are NOT building (resist the scope creep)

- ❌ Proactive "hey, I noticed…" messages
- ❌ Temporal KG / drift / contradiction detection
- ❌ Entity extraction and relationship graph
- ❌ Mood/affect tracking
- ❌ Goal/commitment tracking
- ❌ Multi-user, auth, sessions across users
- ❌ Web UI

When Sub-phase 10.5 is done, Phase 10 is done. Phase 11 (performance + resilience) is a separate, distinct effort.
