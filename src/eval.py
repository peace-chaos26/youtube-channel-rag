# src/eval.py
"""
Lightweight evaluation of the RAG pipeline.

Measures two things:
  1. Retrieval quality  — does the retriever surface the right chunks?
                          Metric: Precision@k
  2. Answer quality     — is the generated answer faithful to the context?
                          Metric: Faithfulness score (via LLM-as-judge)

Why eval matters:
  Without measurement, you can't justify any architectural decision —
  hybrid vs dense, chunk size 256 vs 512, k=4 vs k=8.
  Eval turns opinions into evidence.

Usage:
  python src/eval.py
"""

import json
import logging
from pathlib import Path
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from config import EVAL_DIR, LLM_MODEL
from retriever import HybridRetriever
from chain import QAChain

logger = logging.getLogger(__name__)


# ── Eval dataset ──────────────────────────────────────────────────────────────
# Hand-crafted Q&A pairs where we know the ground truth video + approximate
# timestamp. These are written by you after watching/skimming the videos.
# 10-15 pairs is enough for a portfolio eval set.

EVAL_QUESTIONS = [
    {
        "question": "What does Karpathy say about the role of attention in transformers?",
        "expected_video_id": "kCc8FmEb1nY",          # Let's build GPT
        "expected_keywords": ["attention", "query", "key", "value"],
    },
    {
        "question": "How does Karpathy explain backpropagation through a neural network?",
        "expected_video_id": "VMj-3S1tku0",          # Neural Networks: Zero to Hero
        "expected_keywords": ["gradient", "chain rule", "backward", "derivative"],
    },
    {
        "question": "What is nanoGPT and why did Karpathy build it?",
        "expected_video_id": "kCc8FmEb1nY",
        "expected_keywords": ["nanoGPT", "GPT-2", "reproduce", "clean"],
    },
    {
        "question": "What does Karpathy think about tokenisation in language models?",
        "expected_video_id": "zduSFxRajkE",          # Let's build the GPT Tokenizer
        "expected_keywords": ["token", "BPE", "byte pair", "vocabulary"],
    },
    {
        "question": "How does Karpathy recommend structuring a neural network training loop?",
        "expected_video_id": "VMj-3S1tku0",
        "expected_keywords": ["loss", "optimizer", "forward", "backward", "step"],
    },
]


# ── Metric 1: Retrieval Precision@k ──────────────────────────────────────────

def evaluate_retrieval(
    retriever: HybridRetriever,
    eval_questions: list[dict],
    k: int = 6,
) -> dict:
    """
    Precision@k = (relevant chunks in top-k) / k

    A chunk is "relevant" if it comes from the expected video.
    This is a proxy for true relevance — in a production eval you'd
    have human-labelled relevant chunks per question.

    Returns per-question results and mean Precision@k.
    """
    results = []

    for item in eval_questions:
        question = item["question"]
        expected_video_id = item["expected_video_id"]

        docs = retriever.retrieve(question, k=k)

        # Count how many retrieved chunks come from the expected video
        relevant_count = sum(
            1 for doc in docs
            if doc.metadata.get("video_id") == expected_video_id
        )

        precision_at_k = relevant_count / k

        results.append({
            "question": question,
            "expected_video_id": expected_video_id,
            "relevant_retrieved": relevant_count,
            "k": k,
            "precision_at_k": precision_at_k,
        })

        logger.info(
            f"P@{k}={precision_at_k:.2f} | "
            f"relevant={relevant_count}/{k} | "
            f"{question[:60]}"
        )

    mean_precision = sum(r["precision_at_k"] for r in results) / len(results)

    return {
        "per_question": results,
        "mean_precision_at_k": mean_precision,
        "k": k,
    }


# ── Metric 2: Faithfulness (LLM-as-judge) ─────────────────────────────────────

FAITHFULNESS_PROMPT = PromptTemplate(
    input_variables=["question", "context", "answer"],
    template="""You are evaluating whether an AI-generated answer is faithful
to the provided context. Faithful means: every claim in the answer is
supported by the context. The answer must not introduce information
not present in the context.

Question: {question}

Context:
{context}

Answer:
{answer}

Rate the faithfulness on a scale of 1-5:
  5 = Fully faithful, every claim supported by context
  4 = Mostly faithful, minor unsupported details
  3 = Partially faithful, some claims not in context
  2 = Mostly unfaithful, many unsupported claims
  1 = Completely unfaithful, contradicts or ignores context

Respond with ONLY a JSON object: {{"score": <int>, "reason": "<one sentence>"}}"""
)


def evaluate_faithfulness(
    qa_chain: QAChain,
    eval_questions: list[dict],
    judge_llm: Optional[ChatOpenAI] = None,
) -> dict:
    """
    LLM-as-judge faithfulness evaluation.

    For each question:
      1. Run QAChain to get answer + retrieved context
      2. Ask a judge LLM to score whether the answer is faithful to context
      3. Aggregate scores

    Using a separate judge LLM (or same model in judge role) is standard
    practice — it's the foundation of frameworks like RAGAS.
    """
    judge = judge_llm or ChatOpenAI(model=LLM_MODEL, temperature=0)
    results = []

    for item in eval_questions:
        question = item["question"]

        # Get answer from pipeline
        qa_result = qa_chain.run(question=question)
        answer = qa_result["answer"]
        context = "\n\n".join(
            doc.page_content for doc in qa_result["sources"]
        )

        # Ask judge LLM to score faithfulness
        prompt = FAITHFULNESS_PROMPT.format(
            question=question,
            context=context[:3000],   # truncate context for judge to avoid token overflow
            answer=answer,
        )

        try:
            response = judge.invoke(prompt)
            parsed = json.loads(response.content)
            score = parsed.get("score", 0)
            reason = parsed.get("reason", "")
        except Exception as e:
            logger.error(f"Judge failed for question: {question[:50]} — {e}")
            score = 0
            reason = "eval failed"

        results.append({
            "question": question,
            "answer": answer,
            "faithfulness_score": score,
            "reason": reason,
        })

        logger.info(f"Faithfulness={score}/5 | {question[:60]}")

    mean_faithfulness = sum(r["faithfulness_score"] for r in results) / len(results)

    return {
        "per_question": results,
        "mean_faithfulness": mean_faithfulness,
    }


# ── Run full eval suite ───────────────────────────────────────────────────────

def run_eval(k: int = 6) -> dict:
    """
    Runs retrieval + faithfulness eval and saves results to eval_data/.
    """
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    retriever = HybridRetriever()
    qa_chain = QAChain(retriever=retriever)

    print("\n── Retrieval Eval (Precision@k) ──")
    retrieval_results = evaluate_retrieval(retriever, EVAL_QUESTIONS, k=k)
    print(f"Mean Precision@{k}: {retrieval_results['mean_precision_at_k']:.3f}")

    print("\n── Faithfulness Eval (LLM-as-judge) ──")
    faithfulness_results = evaluate_faithfulness(qa_chain, EVAL_QUESTIONS)
    print(f"Mean Faithfulness: {faithfulness_results['mean_faithfulness']:.2f}/5")

    # Save results
    output = {
        "retrieval": retrieval_results,
        "faithfulness": faithfulness_results,
    }
    output_path = EVAL_DIR / "eval_results.json"
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {output_path}")

    return output


if __name__ == "__main__":
    run_eval(k=6)