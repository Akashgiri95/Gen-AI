# Generative AI

Hands-on coursework and applied projects covering the modern GenAI stack — from how
LLMs tokenize and generate text, through prompt engineering and RAG, to agentic
systems built with LangGraph, multi-agent orchestration with CrewAI, and LLM
evaluation. Built as part of a PGDM in AI & Data Science.

## Learning Path

| # | Topic | What it covers |
|---|-------|-----------------|
| [01](01_llm_fundamentals/) | **LLM Fundamentals** | How transformer LLMs tokenize, encode, and generate text — GPT-2 internals with `transformers` + PyTorch |
| [02](02_prompt_engineering/) | **Prompt Engineering** | Zero-shot, few-shot, and role prompting with Gemini 2.0 Flash |
| [03](03_rag/) | **Retrieval-Augmented Generation** | RAG over structured (CSV) and web-scraped data using LangChain, Chroma, and HuggingFace embeddings |
| [04](04_agentic_langgraph_basics/) | **Agentic AI — LangGraph Basics** | State graphs, chatbot graphs, and tool-using agents (Tavily web search) |
| [05](05_memory_and_hitl/) | **Memory & Human-in-the-Loop** | Persistent checkpointing for multi-turn memory, and pausing agent execution for human approval |
| [06](06_advanced_agents/) | **Advanced Agents** | ReAct agents, dynamic LLM routing via middleware, and LangSmith tracing for observability |
| [07](07_multi_agent_crewai/) | **Multi-Agent Systems (CrewAI)** | A crew of specialized agents collaborating on data analysis and reporting |
| [08](08_llm_evaluation/) | **LLM Evaluation** | Quantitative evaluation — perplexity, ROUGE, BLEU, METEOR, exact match, toxicity, regard |
| [09](09_langchain_agents/) | **LangChain Agents** | Building AI agents with tools, reasoning, multimodal inputs, evaluations, and production guardrails |

## Projects

| Project | Description |
|---------|-------------|
| [LinkedIn Post Generator](projects/linkedin_post_generator/) | Multi-agent (CrewAI) app that mines a resume/PDF for content angles via RAG, then drafts and revises LinkedIn posts with human feedback |
| [Kid's Story Generator](projects/kids_story_generator/) | Streamlit app generating age-appropriate, multi-language stories with illustrations (Gemini Imagen) and narrated audio (gTTS) |
| [Real Estate Research Tool](projects/real_estate_research_tool/) | RAG tool that answers questions over real-estate news articles using LangChain, ChromaDB, and Llama 3 (Groq) |

## Tech Stack

Python · LangChain · LangGraph · CrewAI · Hugging Face Transformers · Sentence-Transformers · ChromaDB ·
FAISS · Gemini / OpenAI / Groq / Llama 3 APIs · Streamlit · `evaluate` (HF Evaluate)

## Key Concepts

Tokenization & autoregressive generation · Prompt engineering (zero-shot, few-shot, CoT, role prompting) ·
Retrieval-Augmented Generation · Vector stores & embeddings · Agentic workflows with LangGraph (state graphs,
tools, conditional edges) · Persistent memory / checkpointing · Human-in-the-loop interrupts · Multi-agent
orchestration · LLM observability (LangSmith) · LLM evaluation metrics (perplexity, ROUGE, BLEU, METEOR,
toxicity, regard)
