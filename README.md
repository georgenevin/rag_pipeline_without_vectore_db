# RAG Pipeline (without a vector DB)

A lightweight Retrieval-Augmented Generation (RAG) pipeline for scraping Infopark news, chunking article text, generating embeddings with Mistral, storing embeddings as JSONL, and answering questions by retrieving and ranking relevant chunks.

This repo is intended as a small, opinionated example showing how to implement a file-based RAG workflow without a dedicated vector database — embeddings are stored in JSONL and ranked with BM25 + cosine similarity.

## Features

- Scrapes Infopark news pages into a single markdown corpus
- Splits markdown into token-aware chunks (tiktoken + langchain-text-splitters)
- Generates embeddings using Mistral's embedding model
- Stores embeddings as newline-delimited JSON (embeddings.jsonl)
- Retrieval via BM25 + cosine similarity (no external vector DB required)
- Evaluation helpers using Langsmith and a Mistral-based judge model

## Stack

- Language: Python 3.12+
- Notable libraries: mistralai, langchain, langchain-text-splitters, langsmith, numpy, tiktoken

## Quickstart

1. Clone the repository

```bash
git clone https://github.com/georgenevin/rag_pipeline_without_vectore_db
cd rag_pipeline_without_vectore_db
```

2. Install dependencies

If you prefer the requirements file:

```bash
pip install -r requirements.txt
```

Or install the package in editable mode:

```bash
pip install -e .
```

3. Provide API credentials

This project uses Mistral for embeddings and judge/chat. Set one of the following environment variables:

- MISTRAL_API_KEY (preferred)
- API_KEY (fallback)

You can add a .env file at the project root with:

```env
MISTRAL_API_KEY=sk-<your-key>
GROQ_API_KEY=<optional-groq-key>
```

Note: The repository currently includes example files and a committed embeddings.jsonl — remove or replace those if you want a fresh run.

4. Scrape the site (writes infopark_news.md)

```bash
python -m rag_pipeline.scraper
```

5. Build embeddings from the markdown corpus

```bash
python -c "from rag_pipeline import build_embeddings_dataset; build_embeddings_dataset()"
```

6. Run a sample query

```bash
python rag.py
```

Or use the package entrypoint:

```bash
python -m rag_pipeline
```

7. Run evaluations (optional, uses Langsmith)

```bash
python -c "from rag_pipeline import eval_caller; eval_caller()"
```

## Configuration

Key configuration lives in rag_pipeline/config.py:

- BASE_URL: Base website (default: https://infopark.in)
- NEWS_INDEX_URL: listing of news
- TARGET_SELECTORS: CSS selectors used to extract title and body (default: ["div.news_title_outer", "div.news_body"]) — update these if the site structure changes.
- CHUNK_SIZE: token-based chunking size used by the text splitter
- EMBEDDINGS_FILE / OUTPUT_FILE: paths for saved artifacts
- get_mistral_api_key() helper that reads MISTRAL_API_KEY or API_KEY from env

## Files of interest

```
rag_pipeline/        main package (scraper, embeddings, retrieval, evaluator)
rag.py               convenience entrypoint and compatibility aliases
infopark_news.md     scraped article corpus (markdown)
news_chunk_1.md      example chunked markdown
embeddings.jsonl     stored embeddings (newline-delimited JSON)
pyproject.toml       package metadata / deps
requirements.txt     install-time dependencies
```

## Notes & recommendations

- API keys are secrets: do not commit real keys. The repo currently expects a `.env` file at the project root for local development; consider adding a `.env.example` with placeholder names.
- The repository includes a committed `embeddings.jsonl`. If you want reproducible runs, either regenerate this file or remove it from Git and add it to .gitignore.
- The scraper relies on CSS selectors; if the Infopark site HTML changes, update TARGET_SELECTORS in rag_pipeline/config.py.
- The current flow stores embeddings in memory as numpy arrays when loaded; large corpora will require a different storage strategy (vector DB) for scaling.

## Contributing

Contributions, issues, and feature requests are welcome. If you'd like me to:

- add a `.env.example`
- remove or regenerate committed embeddings
- add a CI smoke test for the pipeline
- pin dependencies in a lockfile or add GitHub Actions for lint/testing

I can make those changes — tell me which one to do next and I will update the repo.

## License

MIT
