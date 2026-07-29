from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

BASE_URL = "https://infopark.in"
NEWS_INDEX_URL = f"{BASE_URL}/news"
OUTPUT_FILE = PROJECT_ROOT / "infopark_news.md"
CHUNK_FILE_TEMPLATE = PROJECT_ROOT / "news_chunk_1.md"
EMBEDDINGS_FILE = PROJECT_ROOT / "embeddings.jsonl"
CHUNK_SIZE = 5
TARGET_SELECTORS = ["div.news_title_outer", "div.news_body"]
DATASET_NAME = "infopark-rag-eval"
JUDGE_MODEL = "llama-3.1-8b-instant"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def get_mistral_api_key() -> str | None:
    return os.getenv("MISTRAL_API_KEY") or os.getenv("API_KEY")


def get_groq_api_key() -> str | None:
    return os.getenv("GROQ_API_KEY")
