# 09 · LangChain Agents

Build AI agents from scratch using **LangChain** and **Groq** — covering tools, reasoning, multimodal inputs, evaluations, and production guardrails.

## Modules

| # | Folder | What It Builds |
|---|--------|----------------|
| 1 | `1_first_agent` | News reporter agent — searches Google News via SerpAPI and summarises results |
| 2 | `2_agent_with_tools` | Finance agent — fetches live stock prices (yfinance) and private company valuations |
| 3 | `3_reasoning_agent` | Reasoning agent — solves problems step-by-step with no external tools (chain-of-thought) |
| 4 | `4_multimodal_agent` | Image agent — identifies objects from URLs and analyses local clothing images into structured JSON |
| 5 | `5_evals_func` | Evaluation with cosine similarity — tests the inventory agent against a LangSmith dataset |
| 6 | `6_evals_func_llm_judge` | Evaluation with LLM-as-judge — a second model scores the agent's answers |
| 7 | `7_eval_op_metrics` | Operational metrics — reruns the same eval with a different model for head-to-head comparison |
| 8 | `8_guardrails` | Guardrails — PII redaction/masking on outputs, API key blocking on inputs |

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) — fast Python package manager
- API keys (see **Setup** below)

## Setup

**1. Clone and install dependencies**

```bash
git clone https://github.com/Akashgiri95/Gen-AI.git
cd Gen-AI/09_langchain_agents
uv sync
```

**2. Configure API keys**

```bash
cp sample.env .env
```

Open `.env` and fill in your keys:

| Key | Where to get it |
|-----|----------------|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) — free tier available |
| `SERPAPI_API_KEY` | [serpapi.com](https://serpapi.com) — required for module 1 only |
| `HF_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `LANGSMITH_API_KEY` | [smith.langchain.com](https://smith.langchain.com) — required for modules 5–7 |

## Running the Agents

```bash
# Module 1 — News agent
uv run python 1_first_agent/basic_agent.py

# Module 2 — Finance agent
uv run python 2_agent_with_tools/agent_with_tools.py

# Module 3 — Reasoning agent
uv run python 3_reasoning_agent/reasoning_agent.py

# Module 4 — Multimodal (image URL)
uv run python 4_multimodal_agent/multimodal_1.py

# Module 4 — Multimodal (local images → JSON)
uv run python 4_multimodal_agent/multimodal_2.py

# Module 5 — Evaluation: cosine similarity
uv run python 5_evals_func/func_eval.py

# Module 6 — Evaluation: LLM judge
uv run python 6_evals_func_llm_judge/func_eval_llm_judge.py

# Module 7 — Evaluation: operational metrics
uv run python 7_eval_op_metrics/func_eval.py

# Module 8 — Guardrails: PII masking
uv run python 8_guardrails/guardrails_1.py

# Module 8 — Guardrails: API key blocking
uv run python 8_guardrails/guardrails_2.py
```

## Tech Stack

| Tool | Role |
|------|------|
| [LangChain](https://python.langchain.com) | Agent framework — tool binding, message routing, middleware |
| [Groq](https://groq.com) | LLM inference — runs Qwen 32B, LLaMA 4, GPT-oss models at high speed |
| [LangSmith](https://smith.langchain.com) | Tracing and evaluation platform for modules 5–7 |
| [SerpAPI](https://serpapi.com) | Google Search API used in the news agent |
| [yfinance](https://github.com/ranaroussi/yfinance) | Yahoo Finance data for the stock price tool |
| [sentence-transformers](https://sbert.net) | Local embedding model for cosine similarity evaluation |

## Notes

`NOTES.md` contains beginner-friendly course notes with flow diagrams explaining every module — terminology, code walkthroughs, and the mental models behind each concept.
