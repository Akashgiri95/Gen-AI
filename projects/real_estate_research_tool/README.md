# Real Estate Research Tool

A Streamlit RAG app that lets you paste in news article URLs and ask natural-
language questions about their content — with answers grounded in the articles
and source links returned alongside each answer. Built from a Codebasics tutorial
(see [`ATTRIBUTION.md`](ATTRIBUTION.md)) and adapted to use Llama 3 via Groq.

## How it works (`rag.py`)

1. **`process_urls(urls)`** — loads each URL with `UnstructuredURLLoader`, splits
   the content into ~1000-character chunks (`RecursiveCharacterTextSplitter`),
   embeds each chunk with `sentence-transformers/all-MiniLM-L6-v2`
   (`HuggingFaceEmbeddings`), and persists them into a local `Chroma` vector store
   under `resources/vectorstore`.
2. **`generate_answer(query)`** — runs `RetrievalQAWithSourcesChain` against the
   vector store using `ChatGroq` (Llama 3), returning both the answer text and the
   source URLs the answer was drawn from.

## UI (`main.py`)

A minimal Streamlit app: three sidebar inputs for article URLs, a "Process URLs"
button that builds the vector store, and a question box that returns an answer
plus its source articles.

## Run it

```bash
pip install -r requirements.txt
# create a .env with GROQ_MODEL and GROQ_API_KEY
streamlit run main.py
```

## Key Takeaways

- A persistent `Chroma` store (vs. an in-memory one) means the indexed articles
  survive across app restarts — process URLs once, query many times.
- `RetrievalQAWithSourcesChain` returns citations alongside the answer for free —
  an important pattern for any RAG tool where users need to verify claims against
  the original source.
- Swapping the LLM provider (here, Groq/Llama 3 instead of OpenAI/Gemini used
  elsewhere in this repo) required no change to the retrieval pipeline — only the
  `ChatGroq` initialization.

## Tech Stack

Streamlit · LangChain · ChromaDB · HuggingFace Embeddings · Groq (Llama 3)
