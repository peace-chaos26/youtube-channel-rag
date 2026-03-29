# config.py
"""
Central configuration for YouTube Channel RAG.
All tuneable parameters live here — nothing is hardcoded elsewhere.
"""

from dataclasses import dataclass, field
from pathlib import Path


# ── Paths ────────────────────────────────────────────────────────────────────

DATA_DIR = Path("data")
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"     # Raw transcript JSON files
CHROMA_DIR = DATA_DIR / "chroma_db"            # Persisted vector store
EVAL_DIR = Path("eval_data")


# ── Channel ──────────────────────────────────────────────────────────────────

CHANNEL_HANDLE = "@AndrejKarpathy"             # YouTube handle
CHANNEL_ID = "UCXUPKJO5MZQN11PqgIvyuvQ"       # Karpathy's channel ID (stable)
MAX_VIDEOS = None                              # None = ingest entire channel


# ── Chunking ─────────────────────────────────────────────────────────────────

CHUNK_SIZE = 512          # tokens per chunk
CHUNK_OVERLAP = 64        # overlap between consecutive chunks
MIN_CHUNK_LENGTH = 50     # discard chunks shorter than this (noise filter)


# ── Embeddings ───────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "text-embedding-3-small"    # OpenAI
EMBEDDING_DIMENSIONS = 1536


# ── LLM ──────────────────────────────────────────────────────────────────────

LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.0     # Deterministic for factual Q&A
MAX_TOKENS = 1024


# ── Retrieval ─────────────────────────────────────────────────────────────────

TOP_K = 6                 # How many chunks to retrieve per query
RETRIEVAL_MODE = "hybrid" # "dense" | "sparse" | "hybrid"
BM25_WEIGHT = 0.3         # Weight for BM25 in hybrid (dense gets 1 - this)
DENSE_WEIGHT = 0.7


# ── Collection name (ChromaDB) ────────────────────────────────────────────────

CHROMA_COLLECTION = "karpathy_rag"


# ── Metadata fields stored per chunk ─────────────────────────────────────────
# These fields travel with every chunk into the vector store.
# They power timestamp citations and metadata filtering later.

METADATA_FIELDS = [
    "video_id",          # e.g. "kCc8FmEb1nY"
    "video_title",       # e.g. "Let's build GPT: from scratch, in code, spelled out."
    "channel_id",        # stable across renames
    "published_date",    # "2023-01-17"
    "duration_seconds",  # total video length
    "chunk_start_time",  # seconds into video where this chunk begins
    "chunk_end_time",    # seconds into video where this chunk ends
    "url_with_timestamp",# "https://youtube.com/watch?v=kCc8FmEb1nY&t=432"
]