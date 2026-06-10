# 04 · Agentic AI — LangGraph Basics

Introduction to [LangGraph](https://langchain-ai.github.io/langgraph/), the
graph-based framework for building stateful, multi-step LLM applications. Starts
from the simplest possible graph and builds up to a tool-using chatbot.

## Notebooks

### `simple_graph.ipynb` — The Simplest Graph

- Defines a `TypedDict` `State`, three nodes, and conditional edges with
  `Literal` routing and `random` branch selection.
- Compiles and visualizes the graph, then invokes it end-to-end — e.g. starting
  state `{'graph_state': 'Hi, this is Bhavesh.'}` flows through Node 1 → (random
  branch) → Node 3, accumulating state into `'Hi, this is Bhavesh. I am sad!'`.
- Establishes the core LangGraph vocabulary: **State**, **Nodes**, **Edges**,
  **Graph Construction**, **Graph Invocation**.

### `chatbot_graph.ipynb` — Creating a ChatBot using LangGraph

- Builds a `StateGraph` where the message list is the shared state, using the
  prebuilt `add_messages` reducer so new messages append rather than overwrite.
- Wires a single `chatbot` node calling `init_chat_model` (Gemini), compiles the
  graph, and visualizes it with `IPython.display.Image`.
- Runs an interactive loop — user types a question (*"What is Agentic AI?"*), the
  graph responds, and `q` exits the loop.

### `chatbot_with_search_tool.ipynb` — ChatBot with Tavily Search Tool

- Extends the chatbot graph with a **tool**: `TavilySearch`, bound to the LLM via
  `init_chat_model(...).bind_tools()`.
- Adds a `ToolNode` and `tools_condition` conditional edge so the graph routes to
  the search tool only when the LLM decides a query needs live web data.
- Shows the limits of tool-bound agents too — when asked for MOOC recommendations,
  the model correctly defers to the search tool rather than hallucinating course
  names.

## Key Takeaways

- LangGraph models an LLM application as an explicit graph of **state →
  node → state transitions**, which makes multi-step / branching LLM logic
  debuggable and visualizable (vs. a single prompt-and-response call).
- The `add_messages` reducer is what turns a stateless LLM call into a
  conversational chatbot — state persists *within* a graph invocation.
- `tools_condition` + `ToolNode` is the standard LangGraph pattern for "let the
  LLM decide whether to call a tool" — the same pattern used by production agent
  frameworks.

## Tech Stack

LangGraph · LangChain (`init_chat_model`) · Gemini · Tavily Search API
