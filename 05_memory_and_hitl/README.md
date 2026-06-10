# 05 · Memory & Human-in-the-Loop

Two production-critical LangGraph patterns: giving an agent **persistent memory**
across turns, and letting a human **pause, inspect, and approve** an agent's
actions mid-execution.

## Notebooks

### `chatbot_with_memory.ipynb` — Add Memory

- Problem: the tool-using chatbot from [04](../04_agentic_langgraph_basics/) has
  no memory — each invocation starts fresh, so follow-up questions ("remember the
  name of the country I asked about?") fail.
- Solution: compile the graph with an `InMemorySaver` checkpointer and call it
  with a `thread_id`. LangGraph automatically persists state after every step and
  reloads it on the next call with the same `thread_id`.
- Demonstrates a multi-turn conversation where the second question correctly
  references context from the first, and inspects the saved `StateSnapshot`
  history.

### `human_in_the_loop.ipynb` — Human-in-the-Loop

- Adds a custom `human_assistance` tool that calls `interrupt()` — pausing graph
  execution and surfacing a request for human input, ergonomically similar to
  Python's `input()`.
- When the agent calls this tool, execution stops at the `tools` node; the graph
  state shows `('tools',)` as the next step to execute.
- Execution resumes via `Command(resume=...)`, passing the human's response back
  into the paused tool call — the agent then continues as if the tool had returned
  that value.

## Key Takeaways

- **Checkpointing** (`InMemorySaver`, or a persistent backend like SQLite/Postgres
  in production) is what turns a one-shot LLM call into a stateful conversation —
  and the same mechanism enables time-travel debugging and error recovery.
- **`interrupt()` / `Command(resume=...)`** is the LangGraph primitive for
  human-in-the-loop: any node can pause for approval, edits, or extra input
  without restructuring the graph.
- These two patterns compose — a checkpointed graph can be interrupted, inspected,
  and resumed across separate process runs, which is essential for agents that
  take real-world actions (sending emails, executing trades, etc.).

## Tech Stack

LangGraph (`InMemorySaver`, `interrupt`, `Command`) · LangChain · Tavily Search API
