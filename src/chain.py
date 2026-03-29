# src/chain.py
"""
Two chains built on top of the retriever:

  1. QAChain         — answers a specific question using retrieved chunks
  2. SummarizeChain  — summarises all content about a topic across multiple videos
                       using Map-Reduce (handles context window limits)

These are the two query patterns the app exposes:
  - "What does Karpathy say about attention?" → QAChain
  - "Summarise all videos about backpropagation" → SummarizeChain
"""

import logging
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
# from langchain_community.chains.summarize import load_summarize_chain

from config import LLM_MODEL, LLM_TEMPERATURE, MAX_TOKENS, TOP_K
from retriever import HybridRetriever

logger = logging.getLogger(__name__)


# ── LLM instance ──────────────────────────────────────────────────────────────

def get_llm(temperature: float = LLM_TEMPERATURE) -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=temperature,
        max_tokens=MAX_TOKENS,
    )


# ── Prompt templates ──────────────────────────────────────────────────────────

# QA prompt — instructs the model to stay grounded in retrieved context
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an assistant that answers questions based strictly on
Andrej Karpathy's YouTube videos. Use only the provided context to answer.
If the context doesn't contain enough information, say so clearly.
Never fabricate information.

Context:
{context}

Question: {question}

Answer (be specific, cite which video the information comes from):"""
)

# Map prompt — summarises a single chunk during the Map step
MAP_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""Summarise the following excerpt from one of Andrej Karpathy's videos.
Focus on the key technical ideas and concepts discussed.

Excerpt:
{text}

Summary:"""
)

# Reduce prompt — synthesises all per-chunk summaries into a final answer
REDUCE_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""You have been given summaries of multiple excerpts from
Andrej Karpathy's videos, all related to the same topic.

Synthesise these into a single coherent summary. Highlight the most
important concepts, any recurring themes, and key insights.

Summaries:
{text}

Final synthesis:"""
)


# ── Format retrieved chunks into context string ───────────────────────────────

def format_context(documents: list[Document]) -> str:
    """
    Converts retrieved Documents into a structured context string for the LLM.
    Each chunk is labelled with its source video and timestamp.
    This is what makes citations possible in the answer.
    """
    sections = []
    for i, doc in enumerate(documents, 1):
        title = doc.metadata.get("video_title", "Unknown")
        timestamp = doc.metadata.get("chunk_start_time", 0)
        url = doc.metadata.get("url_with_timestamp", "")

        section = (
            f"[Source {i}] {title} @ {int(timestamp)}s\n"
            f"URL: {url}\n"
            f"{doc.page_content}"
        )
        sections.append(section)

    return "\n\n---\n\n".join(sections)


# ── Chain 1: QA Chain ─────────────────────────────────────────────────────────

class QAChain:
    """
    Retrieves relevant chunks and answers a specific question.
    Returns the answer text + the source documents used.

    Pattern: Retrieve → Format context → Single LLM call → Answer
    """

    def __init__(self, retriever: Optional[HybridRetriever] = None):
        self.retriever = retriever or HybridRetriever()
        self.llm = get_llm()

    def run(
        self,
        question: str,
        k: int = TOP_K,
        filter_video_id: Optional[str] = None,
    ) -> dict:
        """
        Args:
            question: user's natural language question
            k: number of chunks to retrieve
            filter_video_id: restrict retrieval to a specific video

        Returns:
            {
                "answer": str,
                "sources": list[Document],   # for rendering citations in UI
            }
        """
        # Step 1: retrieve relevant chunks
        docs = self.retriever.retrieve(
            query=question,
            k=k,
            filter_video_id=filter_video_id,
        )

        if not docs:
            return {
                "answer": "No relevant content found for this question.",
                "sources": [],
            }

        # Step 2: format into context string with source labels
        context = format_context(docs)

        # Step 3: single LLM call with context + question
        prompt = QA_PROMPT.format(context=context, question=question)
        response = self.llm.invoke(prompt)

        return {
            "answer": response.content,
            "sources": docs,
        }


# ── Chain 2: Summarize Chain (Map-Reduce) ─────────────────────────────────────

class SummarizeChain:
    """
    Summarises all content about a topic across multiple videos.
    Implements Map-Reduce manually:
      Map    — summarise each chunk independently
      Reduce — synthesise all summaries into one final answer
    """

    def __init__(self, retriever=None):
        self.retriever = retriever or HybridRetriever()
        self.llm = get_llm()

    def _map(self, doc: Document) -> str:
        """Summarise a single chunk."""
        prompt = MAP_PROMPT.format(text=doc.page_content)
        response = self.llm.invoke(prompt)
        return response.content

    def _reduce(self, summaries: list[str]) -> str:
        """Synthesise all chunk summaries into a final answer."""
        combined = "\n\n---\n\n".join(summaries)
        prompt = REDUCE_PROMPT.format(text=combined)
        response = self.llm.invoke(prompt)
        return response.content

    def run(
        self,
        topic: str,
        k: int = 20,
        filter_video_id: Optional[str] = None,
    ) -> dict:
        docs = self.retriever.retrieve(
            query=topic,
            k=k,
            filter_video_id=filter_video_id,
        )

        if not docs:
            return {
                "summary": "No relevant content found for this topic.",
                "sources": [],
                "video_count": 0,
            }

        # Map step — summarise each chunk independently
        summaries = []
        for doc in docs:
            summary = self._map(doc)
            summaries.append(summary)

        # Reduce step — synthesise all summaries
        final_summary = self._reduce(summaries)

        video_ids = {doc.metadata.get("video_id") for doc in docs}

        return {
            "summary": final_summary,
            "sources": docs,
            "video_count": len(video_ids),
        }


# ── Entry point (smoke test) ──────────────────────────────────────────────────

if __name__ == "__main__":
    # Test QA
    qa = QAChain()
    result = qa.run("What does Karpathy say about attention mechanisms?")
    print("=== QA RESULT ===")
    print(result["answer"])
    print(f"\nSources used: {len(result['sources'])}")
    for doc in result["sources"]:
        print(f"  - {doc.metadata['video_title'][:50]} @ {doc.metadata['url_with_timestamp']}")

    print("\n=== SUMMARIZE RESULT ===")
    summarizer = SummarizeChain()
    result = summarizer.run("backpropagation")
    print(result["summary"])
    print(f"\nCovered {result['video_count']} distinct videos")