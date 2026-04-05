# commons — Reusable LangGraph Infrastructure

A standalone Python package providing production-grade cross-cutting concerns for LangGraph applications. Zero domain dependencies — drop it into any LangGraph project.

## Dependencies

```
pydantic>=2.0
langgraph>=0.2
langchain-core>=0.3
langchain-openai>=0.2   # only needed if using OpenAICallLLM
```

## Package Structure

```
commons/
├── __init__.py              # Re-exports everything for flat imports
├── middleware/              # Node middleware chain + InstrumentedGraph
│   ├── base.py              # NodeMiddleware ABC
│   ├── instrumented_graph.py# StateGraph subclass that auto-wraps nodes
│   ├── logging_mw.py        # Structured entry/exit/error logging
│   ├── metrics_mw.py        # Per-node call counts, durations, errors
│   ├── validation_mw.py     # Pydantic schema validation before node runs
│   ├── error_handling_mw.py # Catch exceptions, convert to NodeResult
│   ├── retry_mw.py          # Exponential backoff retry
│   ├── circuit_breaker_mw.py# Three-state circuit breaker (closed/open/half-open)
│   └── config_mw.py         # Per-node config injection
├── tool_client/             # Transport-agnostic tool framework (MCP-ready)
│   ├── spec.py              # ToolSpec — immutable tool contract
│   ├── registry.py          # ToolRegistry — central tool catalog
│   ├── client.py            # ToolClient ABC + LocalToolClient
│   ├── envelope.py          # ToolResultEnvelope with provenance metadata
│   └── langchain_bridge.py  # Convert ToolSpecs to LangChain function-calling format
├── llm/                     # Protocol-based LLM interaction
│   ├── protocols.py         # CallLLM protocol + LLMRequest/LLMResponse
│   ├── stub.py              # Deterministic stub (no API key needed)
│   ├── openai_adapter.py    # OpenAI/ChatOpenAI adapter
│   ├── quiz.py              # QuizQuestion protocol + simple implementations
│   └── validator.py         # LLMValidator — quiz an LLM before trusting it
├── human/                   # Protocol-based human interaction
│   ├── protocols.py         # CallHuman protocol + HumanRequest/HumanResponse
│   ├── console.py           # CLI-based implementation (input/print)
│   └── mock.py              # Deterministic mock for testing
└── node_validation/         # Structured result envelope for nodes
    ├── result_schema.py     # NodeResult + NodeError
    ├── validator_decorator.py # @validated_node decorator
    └── handle_error.py      # Generic error-handler node
```

---

## Quick Start

### 1. InstrumentedGraph + Middleware

The core value: wrap every LangGraph node with composable cross-cutting concerns.

```python
from commons.middleware import (
    InstrumentedGraph,
    LoggingMiddleware,
    MetricsMiddleware,
    RetryMiddleware,
    ErrorHandlingMiddleware,
)

# Build a graph with middleware
graph = InstrumentedGraph(
    MyState,
    node_middleware=[
        LoggingMiddleware(),                          # outermost: logs entry/exit/errors
        MetricsMiddleware(),                          # tracks call counts + durations
        ErrorHandlingMiddleware(),                    # catches exceptions → NodeResult
        RetryMiddleware(nodes={"call_llm"}, max_retries=2),  # retry only LLM nodes
    ],
)

graph.add_node("validate", validate_fn)   # automatically wrapped
graph.add_node("call_llm", llm_fn)        # automatically wrapped
graph.add_edge(START, "validate")
graph.add_edge("validate", "call_llm")
compiled = graph.compile()
```

### 2. Tool Client

Define tools with Pydantic I/O models. The client validates inputs and outputs, wraps results in metadata envelopes, and is MCP-compatible.

```python
from pydantic import BaseModel, Field
from commons.tool_client import ToolSpec, ToolRegistry, LocalToolClient

class GreetInput(BaseModel):
    name: str = Field(description="Name to greet")

class GreetOutput(BaseModel):
    message: str

def greet_handler(input_: GreetInput) -> GreetOutput:
    return GreetOutput(message=f"Hello, {input_.name}!")

# Register and use
registry = ToolRegistry()
registry.register(ToolSpec(
    name="greet",
    description="Greet someone by name",
    input_model=GreetInput,
    output_model=GreetOutput,
    handler=greet_handler,
))

client = LocalToolClient(registry)
envelope = client.call("greet", {"name": "Alice"})
# envelope.structured == {"message": "Hello, Alice!"}
# envelope.meta.tool_name == "greet"
# envelope.meta.duration_ms == 0.123
```

#### LangChain Integration

```python
from commons.tool_client import specs_to_langchain_tools, execute_tool_call

# Bind tools to a ChatOpenAI model
tool_schemas = specs_to_langchain_tools(client)
llm_with_tools = chat_model.bind_tools(tool_schemas)

# Execute tool calls from LLM responses
for tool_call in ai_response.tool_calls:
    result_text = execute_tool_call(client, tool_call)
```

### 3. LLM Protocol

Swap LLM backends without changing graph code.

```python
from commons.llm import CallLLM, LLMRequest, LLMResponse, call_llm_stub, make_openai_llm

# Testing: deterministic stub
response = call_llm_stub(LLMRequest(
    system_prompt="You are helpful.",
    user_message="Hello",
))

# Production: OpenAI
llm = make_openai_llm(model="gpt-4o-mini")
response = llm(LLMRequest(
    system_prompt="You are helpful.",
    user_message="Hello",
))
```

#### Pre-Run LLM Validation

Quiz an LLM before trusting it with your domain:

```python
from commons.llm import LLMValidator, FactualQuizQuestion, quiz_report_summary

quiz = [
    FactualQuizQuestion(
        question="What are the main components of a REST API?",
        expected_answer="endpoint, method, request, response, status code",
    ),
]

validator = LLMValidator(llm=my_llm, system_prompt=MY_PROMPT, quiz=quiz)
report = validator.run()

if not report.passed:
    print(quiz_report_summary(report))
```

### 4. Human Protocol

Swap human interaction surfaces without changing graph code.

```python
from commons.human import ConsoleHuman, MockHuman, HumanRequest

# CLI development
human = ConsoleHuman(prompt_prefix="You> ")

# Testing
human = MockHuman(responses=["yes", "I fixed it", "done"])

# Use in a node
response = human(HumanRequest(prompt="What should we do about this finding?"))
```

### 5. Node Validation

Structured error handling for LangGraph nodes.

```python
from commons.node_validation import NodeResult, validated_node
from pydantic import BaseModel

class MyNodeInput(BaseModel):
    model_config = {"extra": "ignore"}
    context: str
    session_id: str

@validated_node(MyNodeInput)
def my_node(inp: MyNodeInput, state: dict) -> dict:
    # inp is already validated
    return {"result": f"Processed {inp.session_id}"}

# If validation fails, returns:
# {"node_result": NodeResult(ok=False, error=NodeError(code="INVALID_INPUT", ...))}
```

---

## Writing Custom Middleware

Implement `NodeMiddleware.__call__`:

```python
from commons.middleware import NodeMiddleware

class RateLimitMiddleware(NodeMiddleware):
    def __init__(self, max_per_second: float = 10.0, **kwargs):
        super().__init__(**kwargs)
        self._interval = 1.0 / max_per_second
        self._last_call = 0.0

    def __call__(self, node_name, state, next_fn):
        if not self.applies_to(node_name):
            return next_fn(state)

        import time
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._interval:
            time.sleep(self._interval - elapsed)
        self._last_call = time.monotonic()
        return next_fn(state)
```

---

## Writing Custom Tools

Define Pydantic I/O models and a handler function:

```python
from pydantic import BaseModel, Field
from commons.tool_client import ToolSpec

class SearchInput(BaseModel):
    query: str = Field(description="Search query")
    max_results: int = Field(default=10, description="Max results")

class SearchOutput(BaseModel):
    results: list[str]
    total: int

def search_handler(input_: SearchInput) -> SearchOutput:
    # Your search logic here
    return SearchOutput(results=["result1"], total=1)

search_tool = ToolSpec(
    name="search",
    description="Search the knowledge base",
    input_model=SearchInput,
    output_model=SearchOutput,
    handler=search_handler,
)
```

---

## Design Principles

- **Protocol over ABC** — structural subtyping for CallLLM, CallHuman, QuizQuestion
- **No domain dependencies** — this package knows nothing about your application
- **Composable** — middleware chain, tool registry, protocol-based injection
- **Testable** — stubs and mocks for every protocol (call_llm_stub, MockHuman)
- **MCP-ready** — tool envelopes and schemas align with the Model Context Protocol
