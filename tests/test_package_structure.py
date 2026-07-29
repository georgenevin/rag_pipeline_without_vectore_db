from pathlib import Path

import importlib


def test_package_modules_import() -> None:
    rag_pipeline = importlib.import_module("rag_pipeline")
    scraper = importlib.import_module("rag_pipeline.scraper")
    embeddings = importlib.import_module("rag_pipeline.embeddings")

    assert rag_pipeline.__file__ is not None
    assert hasattr(scraper, "fetch_article_links")
    assert hasattr(embeddings, "token_count")


def test_project_root_is_detected() -> None:
    from rag_pipeline.config import PROJECT_ROOT

    assert Path(PROJECT_ROOT).exists()
    assert (Path(PROJECT_ROOT) / "rag.py").exists()
