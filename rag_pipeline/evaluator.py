from __future__ import annotations

import json
from typing import Any

from langsmith import Client, traceable
from mistralai.client import Mistral

from .config import DATASET_NAME, JUDGE_MODEL, get_mistral_api_key
from .retrieval import query


def get_or_create_dataset() -> Any:
    client = Client()
    existing = list(client.list_datasets(dataset_name=DATASET_NAME))
    if existing:
        print(f"Using existing dataset: {DATASET_NAME}")
        return existing[0]

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Eval set for Infopark news RAG pipeline",
    )

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
        {
            "inputs": {"question": "What are the different Infopark Kochi Phase 2 projects announced in 2025, and who are they with?"},
            "outputs": {"answer": "In 2025, Infopark Kochi Phase 2 saw the Lulu Group announce a ₹500 crore IT tower project (with Lulu IT Twin Towers already inaugurated), and separately the Geojit Tower had its foundation stone laid."},
        },
        {
            "inputs": {"question": "What expansion announcements has Infopark made in 2025 involving GCDA or new phases?"},
            "outputs": {"answer": "In September 2025, Infopark and GCDA signed an MoU for Infopark Phase 3, a 300+ acre 'Integrated AI Township' in Ernakulam district; Infopark also announced Phase 3 and Phase 4 expansions that same month."},
        },
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


def target(inputs: dict[str, str]) -> dict[str, str]:
    question = inputs["question"]
    result = query(question)
    return {"answer": result["answer"], "context": result["context"]}


@traceable(name="groq-faithfulness-judge", run_type="llm")
def run_judge(prompt: str) -> dict[str, Any]:
    api_key = get_mistral_api_key()
    if not api_key:
        raise ValueError("MISTRAL_API_KEY or API_KEY environment variable is not set")

    judge_client = Mistral(api_key=api_key)
    response = judge_client.chat.complete(
        model="mistral-small-2603",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )

    result = json.loads(response.choices[0].message.content)
    if "score" not in result:
        raise ValueError("Judge response does not contain 'score'")

    score = float(result["score"])
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"Judge returned invalid score: {score}. Score must be between 0 and 1.")

    result["score"] = score
    return result


def faithfulness_evaluator(inputs: dict[str, str], outputs: dict[str, str]) -> dict[str, Any]:
    answer = outputs["answer"]
    context = outputs["context"]

    prompt = f"""
You are evaluating the faithfulness of a RAG-generated answer.

Generated answer:
{answer}

Retrieved context:
{context}

Evaluate how well the factual claims in the generated answer are supported by the retrieved context.

Return ONLY valid JSON:

{{
    \"reasoning\": \"Brief explanation\",
    \"score\": 0.0
}}

Score must be between 0.0 and 1.0.
"""
    result = run_judge(prompt)
    return {
        "key": "faithfulness",
        "score": float(result["score"]),
        "comment": result["reasoning"],
    }


def correctness_evaluator(inputs: dict[str, str], outputs: dict[str, str], reference_outputs: dict[str, str]) -> dict[str, Any]:
    question = inputs["question"]
    reference_answer = reference_outputs["answer"]
    answer = outputs["answer"]

    prompt = f"""
You are evaluating the correctness of a RAG-generated answer.

Question:
{question}

Reference answer:
{reference_answer}

Generated answer:
{answer}

Evaluate how correctly the generated answer answers the question compared with the reference answer.

Return ONLY valid JSON:

{{
    \"reasoning\": \"Brief explanation\",
    \"score\": 0.0
}}

The score MUST be a number between 0.0 and 1.0.
"""
    result = run_judge(prompt)
    return {
        "key": "answer_correctness",
        "score": float(result["score"]),
        "comment": result["reasoning"],
    }


def context_relevance_evaluator(inputs: dict[str, str], outputs: dict[str, str]) -> dict[str, Any]:
    question = inputs["question"]
    context = outputs["context"]

    prompt = f"""
You are evaluating retrieval relevance in a RAG system.

Question:
{question}

Retrieved context:
{context}

Evaluate how relevant the retrieved context is for answering the question.

Return ONLY valid JSON:

{{
    \"reasoning\": \"Brief explanation\",
    \"score\": 0.0
}}

The score MUST be between 0.0 and 1.0.
"""
    result = run_judge(prompt)
    return {
        "key": "context_relevance",
        "score": float(result["score"]),
        "comment": result["reasoning"],
    }


def context_recall_evaluator(inputs: dict[str, str], outputs: dict[str, str], reference_outputs: dict[str, str]) -> dict[str, Any]:
    question = inputs["question"]
    reference_answer = reference_outputs["answer"]
    context = outputs["context"]

    prompt = f"""
You are evaluating retrieval recall in a RAG system.

Question:
{question}

Reference answer:
{reference_answer}

Retrieved context:
{context}

Evaluate how much of the information necessary to produce the reference answer is present in the retrieved context.

Return ONLY valid JSON:

{{
    \"reasoning\": \"Brief explanation\",
    \"score\": 0.0
}}

The score MUST be between 0.0 and 1.0.
"""
    result = run_judge(prompt)
    return {
        "key": "context_recall",
        "score": float(result["score"]),
        "comment": result["reasoning"],
    }


def eval_caller() -> Any:
    client = Client()
    experiment_results = client.evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[
            faithfulness_evaluator,
            context_recall_evaluator,
            context_relevance_evaluator,
            correctness_evaluator,
        ],
        experiment_prefix="infopark-rag-eval",
        max_concurrency=2,
    )
    return experiment_results
