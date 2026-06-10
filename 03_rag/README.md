# 03 · Retrieval-Augmented Generation (RAG)

Two RAG pipelines that ground an LLM's answers in external data — one over a
structured CSV catalog, one over content scraped live from web pages — using
LangChain, Chroma as the vector store, HuggingFace embeddings, and Gemini as the
generator.

## Notebooks

### `rag_csv_langchain.ipynb` — RAG over a product catalog

- Loads `ClothingCatalog.csv` with `CSVLoader`, where each row becomes a document.
- Splits/embeds rows with `HuggingFaceEmbeddings` and indexes them in `Chroma`.
- Builds a `RetrievalQA` chain (LangChain) backed by `ChatGoogleGenerativeAI`
  (Gemini) with a custom `PromptTemplate`.
- Demonstrates natural-language product search — e.g. asking for "lightweight
  options" returns specific catalog items with their actual attributes (*"Serene
  Sky 45 Pack (under 2 lbs), Wayfarer Lantern (5.5 oz), Mountain Splendor
  Colorblock Anorak..."*) rather than hallucinated products.

### `rag_chatbot_dementia_care.ipynb` — RAG chatbot over web content

- Uses `SeleniumURLLoader` to scrape live web pages on dementia care.
- Splits documents with `RecursiveCharacterTextSplitter`, embeds with
  HuggingFace embeddings, and indexes in `Chroma`.
- Answers caregiver questions (e.g. *"What is dementia?"*) by retrieving the most
  relevant chunks and passing them to Gemini via a `RetrievalQA` chain — grounding
  the answer in the scraped source material instead of the model's parametric
  knowledge.

## Data

- `ClothingCatalog.csv` — sample product catalog used as the retrieval corpus for
  the first notebook.

## Key Takeaways

- RAG separates "what the model knows" from "what it can look up" — the LLM never
  needs to memorize the catalog or the dementia-care content; it just needs to
  retrieve and reason over it.
- Chunking strategy (`RecursiveCharacterTextSplitter`) and embedding choice
  directly affect retrieval quality — too-large chunks dilute relevance, too-small
  chunks lose context.
- Swapping the data source (CSV vs. live web pages) requires changing only the
  loader — the retrieval + generation pipeline stays identical, which is the core
  appeal of LangChain's abstraction.

## Tech Stack

LangChain · ChromaDB · HuggingFace Embeddings · Gemini (`ChatGoogleGenerativeAI`) ·
Selenium (web scraping)
