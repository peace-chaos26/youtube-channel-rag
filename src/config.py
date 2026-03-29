# config.py

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

DATA_DIR = Path("data")
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"   # subdirs per channel_id
CHROMA_DIR = DATA_DIR / "chroma_db"          # subdirs per channel_id
EVAL_DIR = Path("eval_data")

# ── Channel (runtime — set dynamically, not hardcoded) ───────────────────────

VIDEO_COUNT_WARNING_THRESHOLD = 30           # warn user if channel has more videos

# ── Chunking ─────────────────────────────────────────────────────────────────

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
MIN_CHUNK_LENGTH = 100

# ── Embeddings ───────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# ── LLM ──────────────────────────────────────────────────────────────────────

LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.0
MAX_TOKENS = 1024

# ── Retrieval ─────────────────────────────────────────────────────────────────

TOP_K = 6
RETRIEVAL_MODE = "hybrid"
BM25_WEIGHT = 0.3
DENSE_WEIGHT = 0.7

# ── Metadata fields ───────────────────────────────────────────────────────────

METADATA_FIELDS = [
    "video_id",
    "video_title",
    "channel_id",
    "published_date",
    "duration_seconds",
    "chunk_start_time",
    "chunk_end_time",
    "url_with_timestamp",
]

# ── Per-channel path helpers ──────────────────────────────────────────────────

def get_transcripts_dir(channel_id: str) -> Path:
    """Each channel gets its own transcript subdirectory."""
    return TRANSCRIPTS_DIR / channel_id

def get_chroma_dir(channel_id: str) -> Path:
    """Each channel gets its own ChromaDB subdirectory."""
    return CHROMA_DIR / channel_id

def get_collection_name(channel_id: str) -> str:
    """ChromaDB collection name — must be alphanumeric + underscores."""
    return f"rag_{channel_id.lower()}"