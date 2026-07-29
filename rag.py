from bs4 import BeautifulSoup, NavigableString, Comment
from urllib.parse import urljoin
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq
from langchain_groq import ChatGroq
from mistralai.client import Mistral
import requests
import re
import json
import tiktoken
from typing import List, Dict
import os
import numpy as np
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv
from langsmith import Client
from langsmith import traceable




load_dotenv()

BASE_URL = "https://infopark.in"
NEWS_INDEX_URL = urljoin(BASE_URL, "/news")
OUTPUT_FILE = "infopark_news.md"
CHUNK_SIZE = 5
CHUNK_FILE_TEMPLATE = "news_chunk_1.md"
TARGET_SELECTORS = ["div.news_title_outer", "div.news_body"]
DATASET_NAME = "infopark-rag-eval"
JUDGE_MODEL = "llama-3.1-8b-instant"
groq_api_key = os.getenv("GROQ_API_KEY")
mistral_api_key = os.getenv("MISTRAL_API_KEY")

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
    client = Mistral(api_key=mistral_api_key)
    
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
    api_key = mistral_api_key
    if not api_key:
        raise ValueError("GROY_API_KEY environment variable not set")
    
    client = Mistral(api_key=api_key)
    
    # Create the message with context
    user_message = f"""You are a helpful AI assistant answering questions about Infopark news and developments.

Based on the following context from recent news articles, please answer the user's question.

Context:
{context}

Question: {query_text}

Please provide a clear and concise answer based on the context provided."""
    
    
    response =   client.chat.complete(
        model="mistral-large-latest",
        messages=[
            {
                "role": "user",
                "content": user_message
            }
        ]
    )

  
    answer_text = response.choices[0].message.content
    return {"answer": answer_text, "context": context}


def get_or_create_dataset():
    client = Client()
    existing = list(client.list_datasets(dataset_name=DATASET_NAME))
    if existing:
        print(f"Using existing dataset: {DATASET_NAME}")
        return existing[0]
 
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Eval set for Infopark news RAG pipeline",
    )
 
    # Replace with real Q/A pairs grounded in your scraped articles (infopark_news.md).
    examples = [
        
        {
            "inputs": {"question": "How much is Lulu Group investing in Infopark Kochi Phase 2, and who announced it?"},
            "outputs": {"answer": "LuLu Group chairman M.A. Yusuff Ali will invest ₹500 crore in Infopark Kochi Phase 2, announced by Kerala Chief Minister Pinarayi Vijayan at the inauguration of the Lulu IT Twin Towers."},
        },
        {
            "inputs": {"question": "How big is the proposed IT tower for Lulu Group's Infopark Kochi Phase 2 project, and how many jobs will it create?"},
            "outputs": {"answer": "The proposed IT tower spans 9.5 lakh sq. ft. over 3.5 acres and is expected to generate employment for at least 7,500 professionals."},
        },
        {
            "inputs": {"question": "What is Infopark Phase 3 and how large is the land area involved?"},
            "outputs": {"answer": "Infopark Phase 3 is a planned 'Integrated AI Township' - Kerala's first - spread over more than 300 acres in Ernakulam district, created through a land pooling initiative with GCDA."},
        },
        {
            "inputs": {"question": "Who signed the MoU between Infopark and GCDA for Infopark Phase 3?"},
            "outputs": {"answer": "The MoU was signed by Susanth Kurunthil, CEO of Infopark, and Shary M V, Secretary of GCDA, in the presence of Chief Minister Pinarayi Vijayan."},
        },
        {
            "inputs": {"question": "What recognition did Thrissur receive in the India Skills Report 2025?"},
            "outputs": {"answer": "Thrissur secured a place among the top 5 cities in the Employability of Indian Talent - FY 2025 & Beyond, according to the India Skills Report 2025."},
        },
        {
            "inputs": {"question": "Which Infopark campus is credited with contributing to Thrissur's employability ranking?"},
            "outputs": {"answer": "Infopark Thrissur (Koratty) is credited as a key contributor to the region's improved employability landscape."},
        },
        # --- Synthesis across 2+ articles: tests context recall specifically ---
        {
            "inputs": {"question": "What are the different Infopark Kochi Phase 2 projects announced in 2025, and who are they with?"},
            "outputs": {"answer": "In 2025, Infopark Kochi Phase 2 saw the Lulu Group announce a ₹500 crore IT tower project (with Lulu IT Twin Towers already inaugurated), and separately the Geojit Tower had its foundation stone laid."},
        },
        {
            "inputs": {"question": "What expansion announcements has Infopark made in 2025 involving GCDA or new phases?"},
            "outputs": {"answer": "In September 2025, Infopark and GCDA signed an MoU for Infopark Phase 3, a 300+ acre 'Integrated AI Township' in Ernakulam district; Infopark also announced Phase 3 and Phase 4 expansions that same month."},
        },
        # --- Out-of-corpus / negative control: tests faithfulness (no hallucination) ---
        {
            "inputs": {"question": "What was Infopark's quarterly revenue for Q2 2025?"},
            "outputs": {"answer": "This information is not available in the provided news articles."},
        },
        {
            "inputs": {"question": "Who is the current Prime Minister of India?"},
            "outputs": {"answer": "This information is not covered by the Infopark news corpus - it's outside the scope of what this system is designed to answer."},
        },
    ]
    client.create_examples(dataset_id=dataset.id, examples=examples)
    print(f"Created dataset: {dataset.name} ({len(examples)} examples)")
    return dataset


def target(inputs: dict) -> dict:
    """Wraps the real RAG pipeline so LangSmith evaluates what you actually ship."""
    result = query(inputs["question"])  # now returns {"answer": ..., "context": ...}
    return {"answer": result["answer"], "context": result["context"]}


@traceable(
    name="groq-faithfulness-judge",
    run_type="llm"
)
def run_judge(prompt: str) -> dict:
    # judge_client = Groq()
    judge_client =  Mistral(api_key=mistral_api_key)
    response = judge_client.chat.complete(
        model="mistral-small-2603",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    result = json.loads(response.choices[0].message.content)

    if "score" not in result:
        raise ValueError("Judge response does not contain 'score'")

    score = float(result["score"])

    if not 0.0 <= score <= 1.0:
        raise ValueError(
            f"Judge returned invalid score: {score}. "
            "Score must be between 0 and 1."
        )

    result["score"] = score

    return result



    """Are the retrieved chunks actually relevant to the question?
    Flags a noisy/imprecise retriever even when the final answer happens to be fine.
    """
    evaluator = create_llm_as_judge(
        prompt=RAG_RETRIEVAL_RELEVANCE_PROMPT,
        model=JUDGE_MODEL,
        feedback_key="context_relevance",
  
    )
    return evaluator(
        inputs=inputs,
        outputs=outputs["answer"],
        context=outputs["context"],
    )


def faithfulness_evaluator(inputs: dict, outputs: dict):

    prompt = f"""
You are evaluating the faithfulness of a RAG-generated answer.

Generated answer:
{outputs["answer"]}

Retrieved context:
{outputs["context"]}

Evaluate how well the factual claims in the generated answer
are supported by the retrieved context.

Return ONLY valid JSON:

{{
    "reasoning": "Brief explanation",
    "score": 0.0
}}

Score must be between 0.0 and 1.0.
"""

    result = run_judge(prompt)

    return {
        "key": "faithfulness",
        "score": float(result["score"]),
        "comment": result["reasoning"],
    }
 
def correctness_evaluator(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict
):
    prompt = f"""
You are evaluating the correctness of a RAG-generated answer.

Question:
{inputs["question"]}

Reference answer:
{reference_outputs["answer"]}

Generated answer:
{outputs["answer"]}

Evaluate how correctly the generated answer answers the question
compared with the reference answer.

Return ONLY valid JSON:

{{
    "reasoning": "Brief explanation",
    "score": 0.0
}}

The score MUST be a number between 0.0 and 1.0.

Scoring guidance:
1.0 = Completely correct and complete.
0.75 = Mostly correct with minor missing details.
0.5 = Partially correct.
0.25 = Mostly incorrect but contains some correct information.
0.0 = Completely incorrect.

Do not include any text outside the JSON.
"""

    result = run_judge(prompt)

    return {
        "key": "answer_correctness",
        "score": float(result["score"]),
        "comment": result["reasoning"],
    }
 
def context_relevance_evaluator(inputs: dict, outputs: dict):
    prompt = f"""
You are evaluating retrieval relevance in a RAG system.

Question:
{inputs["question"]}

Retrieved context:
{outputs["context"]}

Evaluate how relevant the retrieved context is for answering
the question.

Return ONLY valid JSON:

{{
    "reasoning": "Brief explanation",
    "score": 0.0
}}

The score MUST be between 0.0 and 1.0.

Scoring guidance:
1.0 = All retrieved information is highly relevant.
0.75 = Most retrieved information is relevant.
0.5 = Some relevant and some irrelevant information.
0.25 = Very little relevant information.
0.0 = Retrieved context is completely irrelevant.

Do not include any text outside the JSON.
"""

    result = run_judge(prompt)

    return {
        "key": "context_relevance",
        "score": float(result["score"]),
        "comment": result["reasoning"],
    }
 
def eval_caller():
    client = Client()
    experiment_results = client.evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[
           
            faithfulness_evaluator,
            context_recall_evaluator,
            context_relevance_evaluator,
            correctness_evaluator
           
          
        ],
        experiment_prefix="infopark-rag-eval",
        max_concurrency=2,
    )
    return experiment_results

def context_recall_evaluator(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict
):
    prompt = f"""
You are evaluating retrieval recall in a RAG system.

Question:
{inputs["question"]}

Reference answer:
{reference_outputs["answer"]}

Retrieved context:
{outputs["context"]}

Evaluate how much of the information necessary to produce the
reference answer is present in the retrieved context.

Return ONLY valid JSON:

{{
    "reasoning": "Brief explanation",
    "score": 0.0
}}

The score MUST be between 0.0 and 1.0.

Scoring guidance:
1.0 = All necessary information is present.
0.75 = Most necessary information is present.
0.5 = About half of the necessary information is present.
0.25 = Only a small amount of necessary information is present.
0.0 = None of the necessary information is present.

Do not include any text outside the JSON.
"""

    result = run_judge(prompt)

    return {
        "key": "context_recall",
        "score": float(result["score"]),
        "comment": result["reasoning"],
    }

if __name__ == "__main__":
    # main()
    #  chunking()
    # result = query("What is Lulu Group's next project in Infopark Kochi Phase 2?")
    # print(result["answer"])
    # LLM Evaluation
      get_or_create_dataset()
      eval_caller()
 