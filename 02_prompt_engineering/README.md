# 02 · Prompt Engineering

Structured experiments in prompt design using Gemini 2.0 Flash via the OpenAI-compatible
API, comparing how different prompting strategies change the quality and shape of LLM
output for the same underlying task.

## Notebook

### `prompt_engineering_techniques.ipynb`

**Phase 1 — Explore**

- **Zero-shot prompting**: ask the model to perform a task (e.g. summarize sales
  performance) with no examples — baseline output quality and format.
- **Few-shot prompting**: provide example input/output pairs before the real
  request, steering the model toward a specific output format/style.
- **Role prompting**: assign the model a persona (e.g. "You are a senior data
  analyst") and observe how tone, vocabulary, and structure shift.

**Phase 2 — Apply**

- Iterates on prompt templates for a business-report-style summarization task,
  comparing zero-shot vs. few-shot output side-by-side — e.g. zero-shot returns a
  generic paragraph, while a few-shot template reliably returns a 3-bullet
  executive summary ("Strong Growth in North America: Quarterly sales increased by
  15%...").

## Key Takeaways

- The same model + same underlying data can produce dramatically different,
  more *usable* output purely based on how the prompt is structured — no
  fine-tuning required.
- Few-shot examples are one of the cheapest, fastest levers for getting consistent
  output format (critical for downstream parsing in production pipelines).
- Role/persona framing is a lightweight way to control tone without changing the
  task instructions.

## Tech Stack

Gemini 2.0 Flash (OpenAI-compatible API) · Python `dotenv` for API key management
