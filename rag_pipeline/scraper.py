from __future__ import annotations

import re
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, Comment, NavigableString

from .config import BASE_URL, CHUNK_FILE_TEMPLATE, HEADERS, NEWS_INDEX_URL, OUTPUT_FILE, TARGET_SELECTORS

INLINE_TAGS = {"strong", "b", "em", "i", "u", "code", "a", "span", "small"}


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def inline_to_markdown(node: Any) -> str:
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


def element_to_markdown(node: Any, indent: int = 0, in_list: bool = False) -> str:
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


def fetch_article_links(index_url: str = NEWS_INDEX_URL) -> list[str]:
    response = requests.get(index_url, headers=HEADERS, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    pattern = re.compile(r"^/news/\d+$")
    links: list[str] = []

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


def format_article_markdown(article: dict[str, str]) -> str:
    source_line = f"Source: [{article['url']}]({article['url']})"
    body = article["markdown"].strip()
    return f"{source_line}\n\n{body}\n"


def scrape_articles(output_path: str | Any = OUTPUT_FILE) -> list[dict[str, str]]:
    links = fetch_article_links()
    articles: list[dict[str, str]] = []

    for idx, link in enumerate(links, start=1):
        print(f"Fetching article {idx}/{len(links)}: {link}")
        markdown = fetch_article_markdown(link)
        articles.append({"url": link, "markdown": markdown})

    if not articles:
        print("No articles found.")
        return articles

    with open(output_path, "w", encoding="utf-8") as file:
        for article in articles:
            file.write(format_article_markdown(article))
            file.write("\n---\n\n")

    print(f"Wrote {output_path} ({len(articles)} articles)")
    return articles


def write_chunks(chunks: list[list[dict[str, str]]]) -> None:
    for index, chunk in enumerate(chunks, start=1):
        chunk_path = CHUNK_FILE_TEMPLATE
        with open(chunk_path, "w", encoding="utf-8") as file:
            for article in chunk:
                file.write(format_article_markdown(article))
                file.write("\n---\n\n")
        print(f"Wrote {chunk_path} ({len(chunk)} articles)")
