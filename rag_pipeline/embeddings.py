from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from mistralai.client import Mistral

from .config import EMBEDDINGS_FILE, OUTPUT_FILE, get_mistral_api_key

embeddings_data: list[dict[str, Any]] = []
embeddings_vectors: np.ndarray | None = None


def token_count(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def split_markdown_into_chunks(content: str, chunk_size: int = 30) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, length_function=token_count, chunk_overlap=5)
    return splitter.split_text(content)


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    api_key = get_mistral_api_key()
    if not api_key:
        raise ValueError("MISTRAL_API_KEY or API_KEY environment variable is not set")

    client = Mistral(api_key=api_key)
    response = client.embeddings.create(model="mistral-embed", inputs=texts)
    return [item.embedding for item in response.data]


def save_embeddings(chunks: list[str], embeddings: list[list[float]], output_path: str | Path = EMBEDDINGS_FILE) -> None:
    if len(chunks) != len(embeddings):
        raise ValueError("Chunk count and embedding count do not match")

    output_path = Path(output_path)
    with open(output_path, "w", encoding="utf-8") as file:
        for idx, (chunk_text, embedding_vector) in enumerate(zip(chunks, embeddings), start=1):
            record = {
                "id": idx,
                "text": chunk_text,
                "embedding": embedding_vector,
            }
            file.write(json.dumps(record) + "\n")

    print(f"Saved {len(embeddings)} embeddings to {output_path}")


def load_embeddings(embeddings_file: str | Path = EMBEDDINGS_FILE) -> None:
    global embeddings_data, embeddings_vectors

    embeddings_file = Path(embeddings_file)
    if not embeddings_file.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")

    embeddings_data = []
    with open(embeddings_file, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                embeddings_data.append(json.loads(line))

    if embeddings_data:
        embeddings_vectors = np.array([item["embedding"] for item in embeddings_data])
    else:
        raise ValueError("No embeddings found in the file")


def build_embeddings_dataset(markdown_path: str | Path = OUTPUT_FILE, output_path: str | Path = EMBEDDINGS_FILE) -> list[str]:
    markdown_path = Path(markdown_path)
    if not markdown_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {markdown_path}")

    with open(markdown_path, "r", encoding="utf-8") as file:
        content = file.read()

    chunks = split_markdown_into_chunks(content)
    embeddings = generate_embeddings(chunks)
    save_embeddings(chunks, embeddings, output_path=output_path)
    return chunks
