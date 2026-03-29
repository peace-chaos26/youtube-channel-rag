# src/vectorstore.py
"""
Embeds all chunked Documents and stores them in ChromaDB.

Two responsibilities:
  1. build_vectorstore() — run once to embed + persist all chunks
  2. load_vectorstore() — run on every query to load the persisted store

Separating build from load means you pay the embedding cost once,
not on every application startup.
"""

import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from dotenv import load_dotenv
load_dotenv()

from config import (
    CHROMA_DIR,
    CHROMA_COLLECTION,
    EMBEDDING_MODEL,
    TRANSCRIPTS_DIR,
)
from chunker import chunk_all_videos

logger = logging.getLogger(__name__)


# ── Embedding model ───────────────────────────────────────────────────────────

def get_embeddings() -> OpenAIEmbeddings:
    """
    Returns the embedding model instance.
    Centralised here so swapping models (e.g. to a local model) is one-line change.
    """
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


# ── Build: embed all chunks and persist ───────────────────────────────────────

def build_vectorstore(
    transcripts_dir: Path = TRANSCRIPTS_DIR,
    persist_dir: Path = CHROMA_DIR,
    batch_size: int = 100,
) -> Chroma:
    """
    Chunks all transcripts, embeds them in batches, and persists to ChromaDB.

    Args:
        batch_size: number of Documents to embed per API call.
                    OpenAI recommends ≤2048 inputs per request;
                    100 is conservative and avoids rate limit errors.

    Returns:
        The populated Chroma vectorstore instance.
    """
    logger.info("Loading and chunking all transcripts...")
    documents = chunk_all_videos(transcripts_dir)
    logger.info(f"Total chunks to embed: {len(documents)}")

    embeddings = get_embeddings()

    # Embed and store in batches
    # Chroma.from_documents handles persistence automatically when persist_directory is set
    logger.info(f"Embedding {len(documents)} chunks in batches of {batch_size}...")

    # Split into batches
    batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]

    vectorstore = None
    for i, batch in enumerate(batches):
        logger.info(f"  Batch {i+1}/{len(batches)} ({len(batch)} chunks)")

        if vectorstore is None:
            # First batch: create the collection
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                collection_name=CHROMA_COLLECTION,
                persist_directory=str(persist_dir),
            )
        else:
            # Subsequent batches: add to existing collection
            vectorstore.add_documents(batch)

    logger.info(f"Vectorstore built and persisted to: {persist_dir}")
    return vectorstore


# ── Load: connect to existing persisted vectorstore ───────────────────────────

def load_vectorstore(persist_dir: Path = CHROMA_DIR) -> Chroma:
    """
    Loads an already-built vectorstore from disk.
    This is what the app and retriever call on every startup — no re-embedding.

    Raises FileNotFoundError if the vectorstore hasn't been built yet.
    """
    if not persist_dir.exists():
        raise FileNotFoundError(
            f"No vectorstore found at {persist_dir}. "
            f"Run: python src/vectorstore.py"
        )

    embeddings = get_embeddings()

    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    count = vectorstore._collection.count()
    logger.info(f"Loaded vectorstore: {count} chunks from {persist_dir}")
    return vectorstore


# ── Utility: check if vectorstore already exists ──────────────────────────────

def vectorstore_exists(persist_dir: Path = CHROMA_DIR) -> bool:
    return persist_dir.exists() and any(persist_dir.iterdir())


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if vectorstore_exists():
        print("Vectorstore already exists. Delete data/chroma_db/ to rebuild.")
    else:
        print("Building vectorstore — this will call the OpenAI embeddings API...")
        vs = build_vectorstore()
        print(f"Done. Collection: {CHROMA_COLLECTION}")

        # Quick sanity check
        results = vs.similarity_search("attention mechanism", k=3)
        print(f"\nSanity check — top 3 results for 'attention mechanism':")
        for r in results:
            print(f"  [{r.metadata['video_title'][:50]}] {r.page_content[:100]}...")