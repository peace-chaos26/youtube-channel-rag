# src/vectorstore.py

import logging
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# New
from config import EMBEDDING_MODEL, get_chroma_dir, get_collection_name, TRANSCRIPTS_DIR
from chunker import chunk_all_videos

logger = logging.getLogger(__name__)


def get_embeddings():
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def build_vectorstore(channel_id: str, batch_size: int = 100) -> Chroma:
    """Build and persist vectorstore for a specific channel."""
    transcripts_dir = TRANSCRIPTS_DIR / channel_id
    persist_dir = get_chroma_dir(channel_id)
    collection_name = get_collection_name(channel_id)

    logger.info(f"Building vectorstore for channel: {channel_id}")
    documents = chunk_all_videos(transcripts_dir)
    logger.info(f"Total chunks to embed: {len(documents)}")

    embeddings = get_embeddings()
    batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]

    vectorstore = None
    for i, batch in enumerate(batches):
        logger.info(f"  Batch {i+1}/{len(batches)} ({len(batch)} chunks)")
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                collection_name=collection_name,
                persist_directory=str(persist_dir),
            )
        else:
            vectorstore.add_documents(batch)

    logger.info(f"Vectorstore persisted to: {persist_dir}")
    return vectorstore


def load_vectorstore(channel_id: str) -> Chroma:
    """Load persisted vectorstore for a specific channel."""
    persist_dir = get_chroma_dir(channel_id)
    collection_name = get_collection_name(channel_id)

    if not persist_dir.exists():
        raise FileNotFoundError(
            f"No vectorstore for channel {channel_id}. Ingest first."
        )

    embeddings = get_embeddings()
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    count = vectorstore._collection.count()
    logger.info(f"Loaded vectorstore: {count} chunks for channel {channel_id}")
    return vectorstore


def vectorstore_exists(channel_id: str) -> bool:
    """Check if a channel has already been ingested and embedded."""
    persist_dir = get_chroma_dir(channel_id)
    return persist_dir.exists() and any(persist_dir.iterdir())