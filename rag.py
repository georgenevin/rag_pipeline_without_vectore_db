from bs4 import BeautifulSoup, NavigableString, Comment
from urllib.parse import urljoin
from langchain_text_splitters import RecursiveCharacterTextSplitter
from mistralai.client import Mistral
import requests
import re
import json
import tiktoken
from typing import List, Dict
import os
import numpy as np
from langchain_community.retrievers import BM25Retriever
from datasets import Dataset
import traceback
from dotenv import load_dotenv





BASE_URL = "https://infopark.in"
NEWS_INDEX_URL = urljoin(BASE_URL, "/news")
OUTPUT_FILE = "infopark_news.md"
CHUNK_SIZE = 5
CHUNK_FILE_TEMPLATE = "news_chunk_1.md"
TARGET_SELECTORS = ["div.news_title_outer", "div.news_body"]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

INLINE_TAGS = {"strong", "b", "em", "i", "u", "code", "a", "span", "small"}

embeddings_data: List[Dict] = []
embeddings_vectors: np.ndarray = None

load_dotenv()


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def inline_to_markdown(node) -> str:
    if isinstance(node, NavigableString):
        return normalize_text(str(node))

    if node.name == "br":
        return "  \n"

    content = "".join(inline_to_markdown(child) for child in node.children)
    if node.name in {"strong", "b"}:
        return f"**{content}**"
    if node.name in {"em", "i", "u", "small"}:
        return f"*{content}*"
    if node.name == "code":
        return f"`{content}`"
    if node.name == "a":
        href = node.get("href", "")
        if href:
            return f"[{content}]({href})"
        return content
    return content


def element_to_markdown(node, indent: int = 0, in_list: bool = False) -> str:
    if isinstance(node, NavigableString):
        return normalize_text(str(node))

    if node.name in {"script", "style", "noscript", "iframe", "svg", "meta", "link"}:
        return ""

    if node.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        level = int(node.name[1])
        text = inline_to_markdown(node)
        return f"{'#' * level} {text}\n\n" if text else ""

    if node.name == "hr":
        return "---\n\n"

    if node.name in {"ul", "ol"}:
        items = []
        for child in node.find_all("li", recursive=False):
            items.append(element_to_markdown(child, indent, in_list=True))
        return "".join(items) + "\n"

    if node.name == "li":
        marker = "- " if node.parent.name == "ul" else "1. "
        prefix = " " * (indent * 2) + marker
        parts = []
        for child in node.children:
            if getattr(child, "name", None) in {"ul", "ol"}:
                parts.append("\n" + element_to_markdown(child, indent + 1, in_list=True))
            else:
                parts.append(inline_to_markdown(child))
        line = prefix + normalize_text(" ".join(part for part in parts if part)).strip()
        return line + "\n"

    if node.name == "blockquote":
        content = "\n".join(
            f"> {normalize_text(inline_to_markdown(child))}"
            for child in node.children
            if normalize_text(inline_to_markdown(child))
        )
        return f"{content}\n\n"

    if node.name == "pre":
        text = node.get_text("\n", strip=True)
        return "```\n" + text + "\n```\n\n"

    if node.name == "table":
        rows = []
        for row in node.find_all("tr", recursive=False):
            cells = [normalize_text(cell.get_text(" ", strip=True)) for cell in row.find_all(["th", "td"], recursive=False)]
            if cells:
                rows.append(cells)
        if not rows:
            return ""
        header = rows[0]
        divider = ["---"] * len(header)
        table_lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(divider) + " |"]
        for row in rows[1:]:
            table_lines.append("| " + " | ".join(row) + " |")
        return "\n".join(table_lines) + "\n\n"

    if node.name in {"p", "div", "section", "article", "header", "footer", "main", "nav", "aside", "figure", "figcaption"}:
        content = " ".join(element_to_markdown(child, indent, in_list) for child in node.children)
        content = normalize_text(content)
        return content + "\n\n" if content else ""

    if node.name in INLINE_TAGS:
        return inline_to_markdown(node)

    return "".join(element_to_markdown(child, indent, in_list) for child in node.children)


def html_to_markdown(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe", "meta", "link", "svg"]):
        tag.extract()
    for comment in soup.find_all(string=lambda string: isinstance(string, Comment)):
        comment.extract()

    selected_nodes = []
    for selector in TARGET_SELECTORS:
        node = soup.select_one(selector)
        if node is not None:
            selected_nodes.append(node)

    if not selected_nodes:
        selected_nodes = [soup.body or soup]

    markdown = "\n".join(element_to_markdown(node) for node in selected_nodes)
    return markdown.strip() + "\n"


def fetch_article_links(index_url: str) -> list[str]:
    response = requests.get(index_url, headers=HEADERS, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    pattern = re.compile(r"^/news/\d+$")
    links = []

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        if pattern.match(href):
            links.append(urljoin(BASE_URL, href))
        elif href.startswith(BASE_URL + "/news/") and pattern.match(href.replace(BASE_URL, "")):
            links.append(href)

    return sorted(dict.fromkeys(links))


def fetch_article_markdown(article_url: str) -> str:
    response = requests.get(article_url, headers=HEADERS, timeout=20)
    response.raise_for_status()
    return html_to_markdown(response.text)


def chunk_list(items: list, chunk_size: int) -> list[list]:
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def format_article_markdown(article: dict[str, str]) -> str:
    source_line = f"Source: [{article['url']}]({article['url']})"
    body = article["markdown"].strip()
    return f"{source_line}\n\n{body}\n"


def write_chunks(chunks: list[list[dict[str, str]]]) -> None:
    for index, chunk in enumerate(chunks, start=1):
        chunk_path = CHUNK_FILE_TEMPLATE
        with open(chunk_path, "w", encoding="utf-8") as file:
            for article in chunk:
                file.write(format_article_markdown(article))
                file.write("\n---\n\n")
        print(f"Wrote {chunk_path} ({len(chunk)} articles)")


def main() -> None:
  
    links = fetch_article_links(NEWS_INDEX_URL)
   

    articles = []
    for idx, link in enumerate(links, start=1):
        print(f"Fetching article {idx}/{len(links)}: {link}")
        markdown = fetch_article_markdown(link)
        articles.append({"url": link, "markdown": markdown})

    if not articles:
        print("No articles found.")
        return

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        for article in articles:
            file.write(format_article_markdown(article))
            file.write("\n---\n\n")

def token_count(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base") 
    return len(encoding.encode(text))

def chunking() -> None:
    with open(OUTPUT_FILE, "r", encoding="utf-8") as file:
        content = file.read()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=30, length_function=token_count, chunk_overlap=5)
    chunks = text_splitter.split_text(content)
    embeddings = embedding(chunks)
    save_embeddings(chunks, embeddings)

def embedding(texts: list[str]) -> list[list[float]]:
    client = Mistral(api_key=os.getenv("API_KEY"))
    
    response = client.embeddings.create(
        model="mistral-embed",
        inputs=texts
    )
    
    return [item.embedding for item in response.data]
    


def save_embeddings(chunks: list[str], embeddings: list[list[float]], output_path: str = "embeddings.jsonl") -> None:
    if len(chunks) != len(embeddings):
        raise ValueError("Chunk count and embedding count do not match")

    with open(output_path, "w", encoding="utf-8") as file:
        for idx, (chunk_text, embedding_vector) in enumerate(zip(chunks, embeddings), start=1):
            record = {
                "id": idx,
                "text": chunk_text,
                "embedding": embedding_vector,
            }
            file.write(json.dumps(record) + "\n")

    print(f"Saved {len(embeddings)} embeddings to {output_path}")





def load_embeddings(embeddings_file: str = "embeddings.jsonl") -> None:
    """Load embeddings from JSONL file."""
    global embeddings_data, embeddings_vectors
    
    if not os.path.exists(embeddings_file):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
    
    # embeddings_data.clear()  # Clear previous data
    
    with open(embeddings_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                embeddings_data.append(data)
    
    # Convert embeddings to numpy array for efficient similarity computation
    if embeddings_data:
        embeddings_vectors = np.array([
            item['embedding'] for item in embeddings_data
        ])
    else:
        raise ValueError("No embeddings found in the file")
        
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def query(query_text: str, top_k: int = 3) -> str:
    """Query the RAG system with a question."""
    global embeddings_data, embeddings_vectors
    
    # Load embeddings if not already loaded
    if embeddings_vectors is None or len(embeddings_data) == 0:
        load_embeddings()
    
    # Get embedding for the query
    query_embeddings = embedding([query_text])
    query_vector = np.array(query_embeddings[0])  # Extract first embedding
    
    text_results = [doc['text'] for doc in embeddings_data]
    bm25_retriever = BM25Retriever.from_texts(text_results)
    bm25_retriever.k = top_k
    context_parts = []
    
    bm25_results = bm25_retriever.invoke(query_text)
    for doc in bm25_results:
        context_parts.append(f"Document (Relevance: Keyword Search Result):\n{doc.page_content}")
   

    # Calculate similarities
    similarities = []
    for idx, doc_embedding in enumerate(embeddings_vectors):
        similarity = cosine_similarity(query_vector, doc_embedding)
        similarities.append((idx, similarity))
    
    # Get top-k results
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_results = similarities[:top_k]
    
    # Build context from retrieved documents
    
    for idx, score in top_results:
        doc = embeddings_data[idx]
        text = doc.get('text', '')
        context_parts.append(f"Document (Relevance: {score:.2%}):\n{text}")
    
    context = "\n\n".join(context_parts)
    
    # Initialize LLM
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY environment variable not set")
    
    client = Mistral(api_key=api_key)
    
    # Create the message with context
    user_message = f"""You are a helpful AI assistant answering questions about Infopark news and developments.

Based on the following context from recent news articles, please answer the user's question.

Context:
{context}

Question: {query_text}

Please provide a clear and concise answer based on the context provided."""
    
    # Call Mistral API using chat.complete
    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[
            {
                "role": "user",
                "content": user_message
            }
        ]
    )
    
    return response.choices[0].message.content


def evaluate_rag_system(queries: list[str], reference_answers: list[str]) -> Dict[str, float]:
 data_samples = {
      "questions" : [
    "What is Lulu Group's next project in Infopark Kochi Phase 2?",
    "How many jobs are expected to be generated by Lulu Group's proposed IT tower?",
    "Why did Infopark receive appreciation from the Union Finance Ministry in 2025?",
    
],

"contexts" : [
    "At the inauguration of Lulu IT Twin Towers, Chief Minister Pinarayi Vijayan announced that Lulu Group chairman M.A. Yusuff Ali will invest ₹500 crore in Infopark Kochi Phase 2. The proposed IT tower will span 9.5 lakh sq. ft. over 3.5 acres and is expected to generate employment for at least 7,500 professionals.",
    
    "The proposed plan for an upcoming IT tower, spanning 9.5 lakh sq. ft. over 3.5 acres, is expected to generate employment for at least 7,500 professionals.",
    
    "Infopark received appreciation from the Central Board of Indirect Taxes and Customs under the Ministry of Finance for timely filing of returns and prompt GST payments during FY 2024-25.",
    
    
],

"ground_truths" : [
    "Lulu Group plans to invest ₹500 crore in Infopark Kochi Phase 2 by developing a 9.5 lakh sq. ft. IT tower on 3.5 acres, generating at least 7,500 jobs.",
    
    "The project is expected to generate employment for at least 7,500 professionals.",
    
    "Infopark was appreciated for timely filing of returns and prompt GST payments during FY 2024-25.",
    
   
],

"answers" : [
    "Lulu Group's next project in Infopark Kochi Phase 2** is an **IT tower** with an investment of **₹500 crore**, announced by the Chief Minister on **June 30, 2025**.The project will span a significant area within Phase 2, which involved the acquisition of **125 acres** for expansion. ",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    ""
]
    }


if __name__ == "__main__":
    # main()
    #  chunking()
     result = query("What is Lulu Group's next project in Infopark Kochi Phase 2?")
     print(result)