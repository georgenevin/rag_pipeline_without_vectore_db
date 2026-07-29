from __future__ import annotations

import numpy as np
from langchain_community.retrievers import BM25Retriever
from mistralai.client import Mistral

from . import embeddings as embeddings_module
from .config import get_mistral_api_key


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def query(query_text: str, top_k: int = 3) -> dict[str, str]:
    if embeddings_module.embeddings_vectors is None or not embeddings_module.embeddings_data:
        embeddings_module.load_embeddings()

    query_embeddings = embeddings_module.generate_embeddings([query_text])
    query_vector = np.array(query_embeddings[0])

    text_results = [doc["text"] for doc in embeddings_module.embeddings_data]
    bm25_retriever = BM25Retriever.from_texts(text_results)
    bm25_retriever.k = top_k
    bm25_results = bm25_retriever.invoke(query_text)

    context_parts: list[str] = []
    for doc in bm25_results:
        context_parts.append(f"Document (Relevance: Keyword Search Result):\n{doc.page_content}")

    similarities = []
    for idx, doc_embedding in enumerate(embeddings_module.embeddings_vectors):
        similarity = cosine_similarity(query_vector, doc_embedding)
        similarities.append((idx, similarity))

    similarities.sort(key=lambda item: item[1], reverse=True)
    top_results = similarities[:top_k]
    for idx, score in top_results:
        doc = embeddings_module.embeddings_data[idx]
        text = doc.get("text", "")
        context_parts.append(f"Document (Relevance: {score:.2%}):\n{text}")

    context = "\n\n".join(context_parts)

    api_key = get_mistral_api_key()
    if not api_key:
        raise ValueError("MISTRAL_API_KEY or API_KEY environment variable is not set")

    client = Mistral(api_key=api_key)
    user_message = f"""You are a helpful AI assistant answering questions about Infopark news and developments.

Based on the following context from recent news articles, please answer the user's question.

Context:
{context}

Question: {query_text}

Please provide a clear and concise answer based on the context provided."""

    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": user_message}],
    )

    answer_text = response.choices[0].message.content
    return {"answer": answer_text, "context": context}
