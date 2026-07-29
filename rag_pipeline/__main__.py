from __future__ import annotations

import argparse

from .embeddings import build_embeddings_dataset
from .retrieval import query
from .scraper import scrape_articles


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Infopark RAG pipeline")
    parser.add_argument("command", choices=["scrape", "embed", "query"], help="Pipeline step to run")
    parser.add_argument("--question", default="What is the latest Infopark news?", help="Question to answer when using query")
    args = parser.parse_args()

    if args.command == "scrape":
        scrape_articles()
    elif args.command == "embed":
        build_embeddings_dataset()
    elif args.command == "query":
        result = query(args.question)
        print(result["answer"])


if __name__ == "__main__":
    main()
