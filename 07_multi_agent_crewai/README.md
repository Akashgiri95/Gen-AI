# 07 · Multi-Agent Systems (CrewAI)

A multi-agent pipeline built with [CrewAI](https://www.crewai.com/), where a
"crew" of specialized agents collaborates — each handling one stage of an
end-to-end data analysis workflow, with output handed off from agent to agent.

## Notebook

### `data_analysis_crewai.ipynb` — Multi-Agent Data Analysis & Visualization

Defines a 5-agent crew, each with a distinct role, goal, and tool access:

| Agent | Responsibility |
|-------|-----------------|
| **File Intake Agent** | Validates the uploaded CSV (format, missing columns) using `CSVSearchTool` |
| **Data Preprocessing Agent** | Cleans the data — removes nulls, normalizes columns, detects categorical vs. numerical fields |
| **Exploratory Analysis Agent** | Computes descriptive statistics — mean, median, std dev, correlations, distributions |
| **Visualization Agent** | Selects and generates appropriate charts (bar, histogram, scatter) for the findings |
| **Reporting Agent** | Synthesizes all upstream output into a final written report with recommendations |

- Agents are orchestrated via `Crew(process=Process.sequential, ...)`, each `Task`
  consuming the previous agent's output.
- Run on a sample student-marks CSV (`Dummy_CA1_Marks1.csv`, not included), the
  crew produces a full markdown analysis report — *"Analysis Report: CA1 Marks
  Dataset... 1. Introduction... 2. Descriptive Statistics... 3. Visualizations..."*
  — generated entirely by the agent pipeline with no manual analysis.

## Key Takeaways

- CrewAI's agent/task/crew abstraction maps cleanly onto a real analyst workflow —
  each agent is a "specialist" with a narrow role, and the crew is the pipeline.
- `Process.sequential` passes context from one agent's task output directly into
  the next agent's task — the same hand-off pattern used in real BI/reporting
  pipelines, but automated end-to-end by LLMs.
- Compared to a single mega-prompt, splitting responsibilities across agents
  produces more structured, reviewable intermediate output (cleaning → stats →
  charts → report) at the cost of more LLM calls.

## Tech Stack

CrewAI (`Agent`, `Task`, `Crew`, `Process`, `LLM`) · `crewai_tools` (`CSVSearchTool`) ·
OpenAI GPT-4o
