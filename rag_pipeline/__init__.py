"""Structured RAG pipeline package for the Infopark news project."""

from .config import PROJECT_ROOT
from .embeddings import build_embeddings_dataset, load_embeddings, save_embeddings
from .evaluator import eval_caller, get_or_create_dataset
from .retrieval import query
from .scraper import scrape_articles

__all__ = [
    "PROJECT_ROOT",
    "build_embeddings_dataset",
    "eval_caller",
    "get_or_create_dataset",
    "load_embeddings",
    "query",
    "save_embeddings",
    "scrape_articles",
]
