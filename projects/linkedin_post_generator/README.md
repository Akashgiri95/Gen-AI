# LinkedIn Post Generator

A Streamlit app that turns a resume (PDF) into a stream of ready-to-publish
LinkedIn posts — using RAG to mine the resume for content angles, a CrewAI
multi-agent pipeline to draft and revise posts, and an interactive feedback loop
for refinement.

## How it works

1. **`extract_text_from_pdf`** — pulls raw text from an uploaded resume PDF
   (`pymupdf`).
2. **`RAGPipeline`** — chunks the resume text (`RecursiveCharacterTextSplitter`),
   embeds it into a `Chroma` vector store, and retrieves relevant sections to
   surface candidate "topics" (achievements, projects, skills) the user could post
   about.
3. **CrewAI agents**:
   - **Backstory Builder** (`story_agent`) — for each topic, builds a richer
     backstory/context to write from.
   - **LinkedIn Post Writer** (`post_writer_agent`) — drafts the actual post(s),
     respecting a configurable max word count and number of posts.
   - **Interactive Post Reframer** (`revision_agent`) — takes human feedback on a
     draft and rewrites the post accordingly (human-in-the-loop revision).
4. **`PostScheduler`** — handles scheduling metadata (post date/time) for the
   generated content.
5. **`LinkedInPostGenerator`** — top-level orchestrator wiring the RAG pipeline and
   the three agents into a single pipeline: *resume → topics → backstories →
   drafts → human-reviewed final posts*.

## UI (`app.py`)

A Streamlit frontend (LinkedIn-blue themed) where the user uploads a resume,
configures the number of posts and word limits, and reviews/edits each generated
post before scheduling.

## Run it

```bash
pip install -r requirements.txt   # streamlit, pandas, pymupdf, crewai, langchain, chromadb, python-dotenv
# add your LLM API key(s) to a .env file
streamlit run app.py
```

## Key Takeaways

- RAG isn't only for Q&A — here it's used to *mine* a document for content ideas,
  turning unstructured resume text into a list of postable topics.
- The 3-agent split (backstory → draft → revise) mirrors a real content workflow:
  research, write, edit — each stage gets its own prompt and role rather than one
  agent doing everything.
- Building in a revision agent from the start makes human-in-the-loop editing a
  first-class part of the pipeline, not an afterthought.

## Tech Stack

Streamlit · CrewAI · LangChain (text splitting, retrieval, `RetrievalQA`) ·
ChromaDB · PyMuPDF · pandas
