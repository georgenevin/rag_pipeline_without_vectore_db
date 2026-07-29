from __future__ import annotations

from rag_pipeline import PROJECT_ROOT
from rag_pipeline.embeddings import (
    build_embeddings_dataset,
    generate_embeddings,
    load_embeddings as load_embedding_store,
    save_embeddings as save_embedding_store,
)
from rag_pipeline.evaluator import eval_caller, get_or_create_dataset
from rag_pipeline.retrieval import cosine_similarity, query
from rag_pipeline.scraper import scrape_articles

# Legacy compatibility aliases
main = scrape_articles
chunking = lambda: build_embeddings_dataset()
embedding = generate_embeddings
save_embeddings = save_embedding_store
load_embeddings = load_embedding_store

__all__ = [
    "PROJECT_ROOT",
    "main",
    "chunking",
    "query",
    "scrape_articles",
    "build_embeddings_dataset",
    "eval_caller",
    "get_or_create_dataset",
    "embedding",
    "save_embeddings",
    "load_embeddings",
    "cosine_similarity",
]


if __name__ == "__main__":
    sample_query = "What is the latest Infopark news?"
    result = query(sample_query)
    print(result["answer"])
