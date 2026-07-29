# RAG Pipeline Without a Vector Database

A lightweight Retrieval-Augmented Generation (RAG) pipeline that scrapes news articles, chunks and embeds them, and answers questions over them — without using a dedicated vector database. This repo demonstrates a simple, file-based approach that stores embeddings in JSONL and performs retrieval with a hybrid BM25 + cosine-similarity approach.

## Table of contents

- [How it works](#how-it-works)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Usage](#usage)
- [Project structure](#project-structure)
- [Notes & configuration](#notes--configuration)
- [Contributing](#contributing)
- [License](#license)

## How it works

1. Scrape — Fetches article links and content from a target site (the example uses Infopark's news page) and converts HTML into clean Markdown while preserving headings, lists, tables, links, etc.
2. Chunk — Splits the scraped Markdown into token-sized chunks using LangChain's `RecursiveCharacterTextSplitter` and measures tokens with `tiktoken`.
3. Embed — Generates embeddings for each chunk using Mistral AI's `mistral-embed` model.
4. Store — Saves chunk text + embeddings as plain JSON lines (`embeddings.jsonl`) — no external vector store required.
5. Retrieve — At query time, combines BM25 keyword retrieval (`langchain_community.retrievers.BM25Retriever`) with cosine similarity computed directly with NumPy over the loaded embeddings.
6. Generate — Passes the combined retrieved context + question to Mistral's chat model (`mistral-large-latest`) to produce a grounded answer.

## Features

- HTML-to-Markdown scraper with support for headings, lists, tables, blockquotes, and inline formatting
- Token-aware chunking via `tiktoken` + LangChain's recursive splitter
- Embeddings generated via Mistral AI (`mistral-embed`)
- Hybrid retrieval: BM25 (keyword) + cosine similarity (semantic), no vector DB required
- Answer generation via Mistral chat completion (`mistral-large-latest`)
- A simple evaluation scaffold (`evaluate_rag_system`) to compare generated answers against ground truths

## Prerequisites

- Python (see `.python-version`)
- A Mistral AI API key

## Setup

1. Clone the repo:

```bash
git clone https://github.com/georgenevin/rag_pipeline_without_vectore_db.git
cd rag_pipeline_without_vectore_db
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

Alternatively, if using `uv` and the included `uv.lock`:

```bash
uv sync
```

3. Create a `.env` file in the project root with your API key:

```
API_KEY=<your-mistral-api-key>
```

Note: the repo currently includes a `.env` file in the tree. Do not commit real secrets — ensure `.env` is listed in `.gitignore` before committing any changes.

## Usage

The pipeline is implemented in `rag.py` and runs in three stages (each callable from the `if __name__ == "__main__":` block):

1. Scrape articles into Markdown

```python
main()
```

This fetches news article links from the configured `BASE_URL` and writes them to `infopark_news.md` (or an alternate file depending on configuration).

2. Chunk and embed

```python
chunking()
```

This splits the scraped Markdown into token-sized chunks, generates embeddings with the configured embedding model, and saves results to `embeddings.jsonl`.

3. Query the RAG system

```python
result = query("What is Lulu Group's next project in Infopark Kochi Phase 2?")
print(result)
```

This loads `embeddings.jsonl`, retrieves relevant chunks using BM25 + cosine similarity, and generates an answer using the chat model.

You can also run the whole script directly:

```bash
python rag.py
```

## Project structure

```
scrapping/
├── rag.py                  # Thin compatibility entrypoint for the original workflow
├── rag_pipeline/           # Structured package for scraping, embeddings, and retrieval
│   ├── __init__.py
│   ├── __main__.py         # CLI entrypoint: python -m rag_pipeline scrape|embed|query
│   ├── config.py           # Shared constants and environment helpers
│   ├── scraper.py          # HTML-to-Markdown scraping and article formatting
│   ├── embeddings.py       # Chunking, embedding generation, and JSONL persistence
│   └── retrieval.py        # Hybrid BM25 + cosine retrieval and answer generation
├── embeddings.jsonl        # Stored chunk embeddings (generated)
├── infopark_news.md        # Scraped news articles in Markdown (generated)
├── news_chunk_1.md         # Chunked article batch (generated)
├── requirements.txt
├── pyproject.toml
├── uv.lock
├── .python-version
├── WINDOWS_SETUP.md        # Windows-specific setup notes
├── INDEX.md
└── .gitignore
```

## Notes & configuration

- Retrieval blends two signals: BM25 for exact keyword matches and cosine similarity over Mistral embeddings for semantic matches — this gives reasonable recall without a vector database.
- `evaluate_rag_system` provides a starting point for measuring retrieval + generation quality; you can extend scoring logic to your needs.
- The scraper targets `div.news_title_outer` and `div.news_body` selectors for Infopark's site. To target a different site, update `TARGET_SELECTORS` and `BASE_URL` in `rag.py` and adapt selectors as needed.
- If you prefer a different environment variable name for clarity (for example, `MISTRAL_API_KEY`), update your local `.env` and the code that reads it accordingly.

## Contributing

Contributions, issues, and feature requests are welcome. Please open an issue or submit a pull request with a clear description of changes.

## License

MIT
