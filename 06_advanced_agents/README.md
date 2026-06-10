# 06 · Advanced Agents

Patterns for building agents beyond a single fixed model and a single tool:
prebuilt ReAct agents, runtime model routing via middleware, and tracing agent
execution with LangSmith.

## Notebooks

### `langgraph_react_agent.ipynb` — Prebuilt ReAct Agent

- Uses LangGraph's prebuilt `create_react_agent` (and the newer
  `langchain.agents.create_agent`) to spin up a tool-calling agent with minimal
  boilerplate — no manual graph wiring needed.
- Binds a simple `get_weather` tool and confirms the agent correctly calls it for
  both Delhi and San Francisco, returning *"It's always sunny in Delhi!"* /
  *"...in sf!"*.
- Shows the tradeoff vs. the hand-built graphs in [04](../04_agentic_langgraph_basics/):
  prebuilt agents are faster to stand up but expose less of the underlying graph
  for customization.

### `dynamic_llm_routing.ipynb` — Dynamic Model Selection via Middleware

- Defines two models of different capability/cost — `gemini-2.0-flash` ("basic")
  and `gemini-2.5-flash` ("advanced").
- Implements `dynamic_model_selection` using `@wrap_model_call` middleware: the
  agent inspects `len(request.state["messages"])` at runtime and routes to the
  advanced model once a conversation exceeds 10 messages, otherwise uses the
  cheaper basic model.
- Demonstrates that middleware can rewrite the `ModelRequest` (including swapping
  the model entirely) before it reaches the LLM — a pattern used in production for
  cost control and complexity-based routing.

### `langsmith_tracing_and_tools.ipynb` — Tool-Calling Agent + LangSmith Tracing

- Builds a small LangGraph agent with a `get_stock_price` tool (returns prices for
  MSFT, AAPL, AMZN, RIL) and a `tools_condition` routing edge.
- Wraps the graph invocation in `@traceable` (LangSmith), so every run — including
  intermediate tool calls and LLM messages — is logged for inspection.
- Runs a multi-step arithmetic query ("buy 20 AMZN at current price, then 15 MSFT —
  what's the total cost?") and gets a fully reasoned, tool-grounded answer
  (*"...total cost for 20 AMZN stocks is 20 × 150 = 3000... grand total is
  3000 + 3004.5..."*), with the full reasoning trace visible in LangSmith.

## Key Takeaways

- Prebuilt agents (`create_react_agent`) trade flexibility for speed — useful for
  prototypes or simple tool-calling tasks; hand-built graphs are better when the
  control flow itself is part of the product.
- Middleware (`wrap_model_call`) is the LangChain mechanism for cross-cutting
  concerns — model routing, logging, retries — without touching the agent's core
  logic.
- LangSmith tracing turns an opaque agent run into an inspectable call graph,
  which is essential for debugging multi-step tool use and for monitoring cost/
  latency in production.

## Tech Stack

LangGraph · LangChain (`create_agent`, `create_react_agent`, middleware) · Gemini ·
LangSmith
