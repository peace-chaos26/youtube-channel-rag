# src/retriever.py
"""
Hybrid retriever combining BM25 (sparse) + dense (embedding) retrieval.
Results are fused using Reciprocal Rank Fusion (RRF).

Why hybrid?
  - Dense retrieval: good at semantic similarity ("what does he say about
    vanishing gradients" → finds conceptually related chunks)
  - BM25: good at exact keyword matches ("karpathy mentions 'nanoGPT'" →
    finds exact term even if embedding similarity is low)
  - Hybrid: best of both worlds, consistently outperforms either alone
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import logging
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

from config import (
    TOP_K,
    BM25_WEIGHT,
    DENSE_WEIGHT,
    RETRIEVAL_MODE,
    CHROMA_DIR,
)
from vectorstore import load_vectorstore

logger = logging.getLogger(__name__)


# ── BM25 Index ────────────────────────────────────────────────────────────────

class BM25Index:
    """
    Lightweight BM25 index built over all documents in the vectorstore.
    Built once at retriever initialisation, held in memory.

    BM25 (Best Match 25) is a classical keyword-based ranking function —
    it scores documents by term frequency weighted by inverse document frequency.
    """

    def __init__(self, documents: list[Document]):
        self.documents = documents
        # Tokenise by whitespace — simple but effective for English transcript text
        tokenised_corpus = [doc.page_content.lower().split() for doc in documents]
        self.index = BM25Okapi(tokenised_corpus)
        logger.info(f"BM25 index built over {len(documents)} documents")

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        """Returns top-k documents with BM25 scores."""
        tokens = query.lower().split()
        scores = self.index.get_scores(tokens)

        # Pair each document with its score, sort descending
        scored = sorted(
            zip(self.documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return scored[:k]


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    dense_results: list[Document],
    sparse_results: list[tuple[Document, float]],
    k: int = 60,               # RRF constant — controls how much low ranks are penalised
    dense_weight: float = DENSE_WEIGHT,
    sparse_weight: float = BM25_WEIGHT,
) -> list[Document]:
    """
    Combines dense and sparse ranked lists into a single ranked list using RRF.

    RRF score for a document = Σ weight / (k + rank)

    A document appearing at rank 1 in both lists scores higher than
    one appearing at rank 1 in only one list. Documents unique to one
    list still get a score — they're not discarded.

    Args:
        k: RRF constant (typically 60). Higher k = more emphasis on
           documents that appear in both lists vs. top-ranked in one.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    # Score dense results
    for rank, doc in enumerate(dense_results, start=1):
        doc_id = doc.metadata.get("url_with_timestamp", doc.page_content[:50])
        scores[doc_id] = scores.get(doc_id, 0) + dense_weight / (k + rank)
        doc_map[doc_id] = doc

    # Score sparse (BM25) results
    for rank, (doc, _) in enumerate(sparse_results, start=1):
        doc_id = doc.metadata.get("url_with_timestamp", doc.page_content[:50])
        scores[doc_id] = scores.get(doc_id, 0) + sparse_weight / (k + rank)
        doc_map[doc_id] = doc

    # Sort by fused score descending
    ranked_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[doc_id] for doc_id in ranked_ids]


# ── Main Retriever ────────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Retriever that supports three modes:
      - "dense"  : embedding similarity only (ChromaDB)
      - "sparse" : BM25 keyword matching only
      - "hybrid" : RRF fusion of both (default)

    Also supports metadata filtering — e.g. restrict retrieval to a
    specific video or date range.
    """

    def __init__(
        self,
        vectorstore: Optional[Chroma] = None,
        mode: str = RETRIEVAL_MODE,
    ):
        self.mode = mode
        self.vectorstore = vectorstore or load_vectorstore()

        # Load all documents from ChromaDB to build BM25 index
        # ChromaDB stores documents; we retrieve all to index for BM25
        logger.info("Loading all documents for BM25 index...")
        all_docs = self.vectorstore.get()  # returns dict with "documents" and "metadatas"

        self.all_documents = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(all_docs["documents"], all_docs["metadatas"])
        ]

        self.bm25 = BM25Index(self.all_documents)
        logger.info(f"HybridRetriever ready. Mode: {self.mode}")

    def retrieve(
        self,
        query: str,
        k: int = TOP_K,
        filter_video_id: Optional[str] = None,      # restrict to one video
        filter_date_after: Optional[str] = None,    # "YYYYMMDD"
    ) -> list[Document]:
        """
        Main retrieval method. Returns top-k Documents for a query.

        Args:
            query: natural language question
            k: number of chunks to return
            filter_video_id: if set, only retrieve chunks from this video
            filter_date_after: if set, only retrieve chunks from videos
                               published after this date
        """
        # Build ChromaDB metadata filter if requested
        chroma_filter = self._build_filter(filter_video_id, filter_date_after)

        if self.mode == "dense":
            return self._dense_retrieve(query, k, chroma_filter)

        elif self.mode == "sparse":
            results = self.bm25.search(query, k)
            return [doc for doc, _ in results]

        else:  # hybrid (default)
            # Retrieve more than k from each, then fuse and trim to k
            dense_results = self._dense_retrieve(query, k * 2, chroma_filter)
            sparse_results = self.bm25.search(query, k * 2)
            fused = reciprocal_rank_fusion(dense_results, sparse_results)
            return fused[:k]

    def _dense_retrieve(
        self,
        query: str,
        k: int,
        chroma_filter: Optional[dict],
    ) -> list[Document]:
        """Embedding similarity search via ChromaDB."""
        return self.vectorstore.similarity_search(
            query,
            k=k,
            filter=chroma_filter,
        )

    def _build_filter(
        self,
        video_id: Optional[str],
        date_after: Optional[str],
    ) -> Optional[dict]:
        """
        Builds a ChromaDB metadata filter dict.
        ChromaDB uses MongoDB-style filter syntax.
        """
        conditions = []

        if video_id:
            conditions.append({"video_id": {"$eq": video_id}})

        if date_after:
            conditions.append({"published_date": {"$gte": date_after}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}


# ── Entry point (smoke test) ──────────────────────────────────────────────────

if __name__ == "__main__":
    retriever = HybridRetriever(mode="hybrid")

    query = "What does Karpathy say about attention mechanisms?"
    results = retriever.retrieve(query, k=3)

    print(f"\nQuery: {query}\n")
    for i, doc in enumerate(results, 1):
        print(f"[{i}] {doc.metadata['video_title'][:60]}")
        print(f"     ⏱  {doc.metadata['chunk_start_time']:.0f}s  →  {doc.metadata['url_with_timestamp']}")
        print(f"     {doc.page_content[:150]}...")
        print()