# Conversation Engine - Pydantic Models

This package implements the domain models for the **Reusable Conversation Engine** as defined in `design/WHITEBOARD_ONTOLOGY.md`.

## Overview

The Conversation Engine is designed to enable structured planning conversations between humans and AI, transforming ambiguous input into structured planning artifacts through conversation, critique, and governance.

## Design Philosophy

These models follow strong design principles:

- **Strongly typed**: All inputs/outputs use Pydantic models with explicit types
- **Immutable where appropriate**: Rules and edges are frozen; nodes are mutable
- **Clear separation of concerns**: Models organized by layer (nodes, edges, rules, queries, assessments)
- **Graph-native**: Designed for knowledge graph representation with explicit node and edge types
- **AI-first**: Built to support AI reasoning over architectural knowledge

## Model Structure

### Base Models (`models/base.py`)

- **`NodeType`**: Literal type defining all valid node types
- **`EdgeType`**: Literal type defining all valid edge types
- **`BaseNode`**: Base class for all knowledge graph nodes
- **`BaseEdge`**: Base class for all knowledge graph edges

### Node Models (`models/nodes.py`)

Knowledge graph nodes representing architectural artifacts:

- **`Feature`**: High-level system capability
- **`Goal`**: Desired outcome or objective
- **`GuidingPrinciple`**: Design or architectural principle
- **`Requirement`**: Specific system need or constraint
- **`Capability`**: Observable system behavior
- **`UseCase`**: User interaction or workflow
- **`Scenario`**: Concrete instance of a use case
- **`DesignArtifact`**: Design decision or architectural element
- **`Decision`**: Architectural or design decision
- **`Constraint`**: System limitation or restriction
- **`Component`**: System module or architectural component
- **`Dependency`**: External system, library, or service
- **`DocumentationArtifact`**: Documentation or explanatory content

### Traceability Models (`models/traceability.py`)

Explicit traceability chains between layers:

- **`GoalRequirementTrace`**: Goal → Requirements
- **`RequirementCapabilityTrace`**: Requirement → Capabilities
- **`CapabilityComponentTrace`**: Capability → Components
- **`ComponentDependencyTrace`**: Component → Dependencies

### Integrity Rules (`models/rules.py`)

Machine-readable constraints for graph validation:

- **`IntegrityRule`**: Defines structural constraints (e.g., "every goal must have ≥1 requirement")
- **`RuleType`**: Types of validation checks (minimum_outgoing_edge_count, exact_outgoing_edge_count, etc.)
- **`Severity`**: Violation severity levels (low, medium, high)

### Query Patterns (`models/queries.py`)

Reusable AI reasoning tools for graph analysis:

- **`GraphQueryPattern`**: Defines queries for gap detection, completeness checks, impact analysis
- **`EdgeCheck`**: Checks for presence/absence of edges
- **`TraversalSpec`**: Defines graph traversal for impact analysis
- **`PathStep`**: Single step in a lineage trace pattern
- **`QueryIntent`**: Purpose of query (gap_detection, orphan_detection, lineage_trace, etc.)

### Assessment Models (`models/assessment.py`)

Structured output format for AI findings:

- **`Assessment`**: AI-generated finding about graph gaps or inconsistencies
- **`AssessmentType`**: Types of issues (missing_goal_coverage, orphan_decision, etc.)
- **`Confidence`**: AI confidence level (low, medium, high)

## Example Data

The `examples/ontology_data.py` module provides example data from WHITEBOARD_ONTOLOGY.md:

```python
from ogar.examples import (
    get_goals,
    get_requirements,
    get_capabilities,
    get_components,
    get_goal_requirement_traces,
    get_sample_integrity_rules,
    get_sample_query_patterns,
)

# Load all goals from the ontology
goals = get_goals()

# Load traceability data
traces = get_goal_requirement_traces()

# Load integrity rules
rules = get_sample_integrity_rules()
```

## Usage

### Creating Nodes

```python
from conversation_engine.models import Goal, Requirement, Component

goal = Goal(
    id="goal-structured-convergence",
    name="Structured Convergence",
    statement="Enable the system to transform ambiguous human input into structured artifacts."
)

requirement = Requirement(
    id="REQ-001",
    name="Advisory Conversation Support",
    requirement_type="functional"
)

component = Component(
    id="component-artifact-synthesizer",
    name="Artifact Synthesizer",
    description="Synthesizes structured artifacts from conversation"
)
```

### Creating Traceability

```python
from conversation_engine.models import GoalRequirementTrace

trace = GoalRequirementTrace(
    goal_id="goal-structured-convergence",
    requirement_ids=["REQ-001", "REQ-004", "REQ-005"]
)
```

### Defining Integrity Rules

```python
from conversation_engine.models import IntegrityRule

rule = IntegrityRule(
    id="rule-goal-must-have-requirement",
    name="Goal must map to requirement",
    description="Every goal must map to at least one requirement.",
    applies_to_node_type="goal",
    rule_type="minimum_outgoing_edge_count",
    target_node_types=["requirement"],
    minimum_count=1,
    severity="high",
    failure_message_template="Goal '{subject_name}' has no linked requirements."
)
```

### Defining Query Patterns

```python
from conversation_engine.models import GraphQueryPattern, EdgeCheck

query = GraphQueryPattern(
    id="query-missing-goal-coverage",
    name="Missing Goal Coverage",
    description="Find goals that are not linked to any requirements.",
    subject_node_type="goal",
    query_intent="gap_detection",
    checks=[
        EdgeCheck(
            edge_type="SATISFIED_BY",
            target_node_types=["requirement"],
            expected_min_count=1
        )
    ],
    output_kind="finding_set"
)
```

## Conversation Graph (`graph/`)

A **domain-agnostic** LangGraph conversation loop.  All domain logic is
injected via the `ConversationContext` protocol — the loop itself has no
knowledge of graphs, rules, or ontologies.

### Architecture

```
┌──────────────────────────────────────────────────┐
│  ConversationContext (Protocol)                   │
│  ── validate(prior_findings) → ValidationResult  │
│  ── format_finding_summary(findings) → str       │
│  ── get_domain_state() → dict                    │
└────────────────────┬─────────────────────────────┘
                     │  implements
       ┌─────────────┴──────────────┐
       │ ArchitecturalOntologyContext│  ← concrete adapter
       │   KnowledgeGraph           │
       │   IntegrityRule[]          │
       │   RuleEvaluator            │
       └────────────────────────────┘
```

### Current topology

```
START → validate → reason → respond → route
route → validate   (open findings remain + turns < max)
route → END        (all clear, max turns, or error/complete)
```

### Files

- **`graph/context.py`** — `ConversationContext` (Protocol), `Finding`, `ValidationResult`
- **`graph/architectural_context.py`** — `ArchitecturalOntologyContext` (concrete implementation)
- **`graph/state.py`** — `ConversationState` (TypedDict), `ConversationInput`, `ConversationOutput`
- **`graph/nodes.py`** — `validate`, `reason` (stub), `respond` (stub) — all domain-agnostic
- **`graph/builder.py`** — `build_conversation_graph()` + `route_after_respond` router

### Key design decisions

- **Protocol, not ABC** — structural subtyping; any class with the right methods satisfies the contract
- **`Finding` is a frozen dataclass** — domain-agnostic; the loop never sees `Assessment`
- **Nodes never import domain types** — they call `context.validate()` and `context.format_finding_summary()`
- **Domain state is opaque** — `get_domain_state()` returns a dict the loop stores but never inspects

### Key LangGraph concepts used

- **`StateGraph(ConversationState)`** — graph typed by the state schema
- **`add_messages` reducer** — on `messages` field, LangGraph appends instead of replacing
- **Partial state returns** — each node returns only the keys it changes
- **`add_conditional_edges`** — router function returns next node name or `"__end__"`

### Usage (architectural ontology)

```python
from conversation_engine.graph import (
    build_conversation_graph,
    ArchitecturalOntologyContext,
    DomainConfig,
)
from ogar.fixtures import create_graph_with_gaps
from conversation_engine.models.rule_node import IntegrityRule

config = DomainConfig(
    project_name="my-project",
    knowledge_graph=create_graph_with_gaps(),
    rules=[your_rules],
)
context = ArchitecturalOntologyContext(config)
graph = build_conversation_graph()
result = graph.invoke({
    "context": context,
    "session_id": "session-1",
    "findings": [],
    "messages": [],
    "current_turn": 0,
    "status": "running",
})
```

### Usage (custom domain — no architectural imports needed)

```python
from conversation_engine.graph import (
    build_conversation_graph,
    ConversationContext,
    Finding,
    ValidationResult,
)

class MyCodeReviewContext:
    def validate(self, prior_findings):
        # run your own linter / scanner / whatever
        return ValidationResult(findings=[...])

    def format_finding_summary(self, findings):
        return f"{len(findings)} lint issues found."

    def get_domain_state(self):
        return {"files_scanned": 42}

graph = build_conversation_graph()
result = graph.invoke({
    "context": MyCodeReviewContext(),
    "session_id": "review-1",
    "findings": [],
    "messages": [],
    "current_turn": 0,
    "status": "running",
})
```

---

## Research Files (`research/files/`)

These files were produced during a design session with Claude Sonnet. They represent a **more elaborate target architecture** with 7 nodes, an interruption policy, mutation review protocol, and anchored exchange tracking.

They are kept as **design reference only** — not wired into the working code.

### Why we care about them

The research files solve real problems we will face as the conversation engine matures:

- **`AnchoredExchange` + `BeliefChange`** — Track which conversation turns actually changed the knowledge graph. Without this, we can't explain *why* the graph looks the way it does after a multi-turn session.
- **`InterruptionPolicy`** — A compound policy (confidence threshold, assessment type triggers, autonomous turn limits) that decides when to pause and ask the human. Without this, the agent either always asks or never asks.
- **`MutationReviewer`** (ABC + `AutoApproveReviewer` stub) — Interface for reviewing proposed graph mutations before committing them. Separates "what the agent wants to do" from "what actually happens."
- **`integrate` vs `mutate_graph` as separate nodes** — Lets you validate a proposed mutation before committing it to the graph.
- **Router with priority-ordered exits** — Error → interrupted → complete/hand_off → max_turns → continue.

The architecture is sound but front-loads too much design for where we are today. Each concept should be introduced only when the previous layer is working and tested.

---

## Roadmap

| # | Step | Status |
|---|------|--------|
| 1 | Basic loop: validate → reason → respond → route | ✅ Done |
| 2 | Separate concerns: `ConversationContext` protocol + injectable domains | ✅ Done |
| 3 | Swap `reason` stub for real LLM | Next |
| 4 | Wire `GraphQueryPattern` execution as LangGraph tools | Pending |
| 5 | Add `InterruptionPolicy` (pure logic, no LLM) | Pending |
| 6 | Add `integrate` + `mutate_graph` + `MutationReviewer` | Pending |
| 7 | Add `AnchoredExchange` tracking | Pending |

Each step earns its keep when the previous one is working. The research files are a good roadmap — just not code to copy-paste.

---

## Testing

Run the full test suite (110 tests):

```bash
/opt/miniforge3/envs/langgraph-sandbox/bin/python -m pytest tests/conversation_engine/ -v
```

Test coverage:
- **Models** — Node creation, serialization, traceability, rules, queries, assessments
- **Storage** — KnowledgeGraph CRUD, indexes, GraphQueries (neighbors, orphans, paths, coverage)
- **Validation** — RuleEvaluator with all rule types, fixtures, severity filtering
- **Conversation graph** — Node unit tests, router logic, full graph integration (18 tests)
- **Domain agnosticism** — Fake context proves the loop works with arbitrary domains (3 tests)

## Design Rationale

### Why Pydantic over TypedDict?

- **Validation**: Pydantic validates data at runtime
- **Serialization**: Built-in JSON serialization/deserialization
- **Documentation**: Field descriptions become part of the schema
- **Type safety**: Stricter than TypedDict with better IDE support
- **Immutability**: Can freeze models where appropriate (rules, edges)

### Why separate traceability models?

Traceability is explicit and queryable, not implicit in node references. This enables:
- Gap detection (goals with no requirements)
- Coverage analysis (requirements with no capabilities)
- Impact analysis (what components are affected by a decision)
- Lineage tracing (goal → requirement → capability → component)

### Why frozen rules and edges?

- **Rules** are design-time artifacts that should not change at runtime
- **Edges** represent immutable relationships (create new edges instead of modifying)
- **Nodes** are mutable because their content evolves during conversation

## Architecture Alignment

These models directly implement the ontology layers from WHITEBOARD_ONTOLOGY.md:

1. **Intent Layer** → Goal, GuidingPrinciple, Feature
2. **Requirement Layer** → Requirement
3. **Behavior Layer** → Capability, UseCase, Scenario
4. **Design Knowledge Layer** → DesignArtifact, Decision, Constraint
5. **Design Layer** → Component, Dependency, DocumentationArtifact
6. **Traceability Layer** → *Trace models
7. **Integrity Rules Layer** → IntegrityRule
8. **Query Patterns Layer** → GraphQueryPattern
9. **Assessment Layer** → Assessment
