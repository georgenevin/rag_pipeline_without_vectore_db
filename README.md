# RAG Pipeline Without a Vector Database
 
A lightweight Retrieval-Augmented Generation (RAG) pipeline that scrapes news articles, chunks and embeds them, and answers questions over them — **without using a vector database**. Embeddings are stored in a plain JSONL file, and retrieval is done in-memory using a hybrid of **BM25 keyword search** and **cosine similarity** over the embeddings.
 
## How It Works
 
1. **Scrape** — Fetches article links and content from [Infopark's news page](https://infopark.in/news), converting the HTML into clean Markdown (headings, lists, tables, links, etc. preserved).
2. **Chunk** — Splits the scraped Markdown into token-sized chunks using `RecursiveCharacterTextSplitter`, measured with `tiktoken`.
3. **Embed** — Generates embeddings for each chunk using Mistral AI's `mistral-embed` model.
4. **Store** — Saves chunk text + embeddings as plain JSON lines (`embeddings.jsonl`) — no external vector store required.
5. **Retrieve** — At query time, combines:
   - **BM25** keyword-based retrieval (`langchain_community.retrievers.BM25Retriever`)
   - **Cosine similarity** search computed directly with NumPy over the loaded embeddings
6. **Generate** — Passes the combined retrieved context + question to Mistral's `mistral-large-latest` chat model to produce a grounded answer.
## Why No Vector Database?
 
This project demonstrates that for small-to-medium document sets, you don't need a dedicated vector database (Pinecone, Chroma, Weaviate, etc.). Storing embeddings in a JSONL file and computing similarity in-memory with NumPy is simple, dependency-light, and easy to inspect or debug.
 
## Features
 
- HTML-to-Markdown scraper with support for headings, lists, tables, blockquotes, and inline formatting
- Token-aware chunking via `tiktoken` + LangChain's recursive splitter
- Embeddings generated via Mistral AI (`mistral-embed`)
- Hybrid retrieval: BM25 (keyword) + cosine similarity (semantic), no vector DB
- Answer generation via Mistral chat completion (`mistral-large-latest`)
- Simple RAGAS-style evaluation scaffold (`evaluate_rag_system`) with sample questions, contexts, and ground truths
## Prerequisites
 
- Python (see `.python-version`)
- A [Mistral AI](https://mistral.ai) API key
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
   Or, if using [uv](https://github.com/astral-sh/uv) (this repo includes a `uv.lock`):
```bash
   uv sync
```
 
3. Create a `.env` file in the project root:
```
   API_KEY=<your-mistral-api-key>
```
 
   > ⚠️ A `.env` file is currently tracked in this repo — make sure it doesn't contain a real key, and confirm `.env` is listed in `.gitignore` before committing further changes.
 
## Usage
 
The pipeline runs in three independent stages, controlled from the `if __name__ == "__main__":` block in `rag.py`:
 
**1. Scrape articles into Markdown:**
```python
main()
```
Fetches all news article links from Infopark's news index and writes them to `infopark_news.md`.
 
**2. Chunk and embed:**
```python
chunking()
```
Splits `infopark_news.md` into token-sized chunks, embeds them with Mistral, and saves the result to `embeddings.jsonl`.
 
**3. Query the RAG system:**
```python
result = query("What is Lulu Group's next project in Infopark Kochi Phase 2?")
print(result)
```
Loads `embeddings.jsonl`, retrieves relevant chunks via BM25 + cosine similarity, and generates an answer using Mistral's chat model.
 
Run the script directly:
```bash
python rag.py
```
 
## Project Structure
 
```
rag_pipeline_without_vectore_db/
├── rag.py               # Scraper, chunker, embedder, and RAG query pipeline
├── embeddings.jsonl      # Stored chunk embeddings (generated)
├── infopark_news.md      # Scraped news articles in Markdown (generated)
├── news_chunk_1.md        # Chunked article batch (generated)
├── requirements.txt
├── pyproject.toml
├── uv.lock
├── .python-version
├── WINDOWS_SETUP.md       # Windows-specific setup notes
├── INDEX.md
└── .gitignore
```
 
## Notes
 
- Retrieval blends two signals: BM25 for exact keyword matches and cosine similarity over Mistral embeddings for semantic matches — giving reasonable recall without any vector database infrastructure.
- `evaluate_rag_system` sketches out a RAGAS-style evaluation setup (questions, contexts, ground truths, generated answers) for measuring retrieval and generation quality, though the scoring logic itself isn't fully wired up yet.
- The scraper targets `div.news_title_outer` and `div.news_body` selectors specific to Infopark's site structure — adjust `TARGET_SELECTORS` and `BASE_URL` in `rag.py` to scrape a different source.
## License
 
MIT
