# src/chunker.py
"""
Converts raw transcript JSON files into overlapping text chunks
with timestamp metadata preserved per chunk.

This is where transcript segments (short, ~3s each) are merged into
semantically meaningful chunks (~512 tokens) that the LLM can reason over,
while keeping track of where in the video each chunk came from.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_CHUNK_LENGTH,
    TRANSCRIPTS_DIR,
)

logger = logging.getLogger(__name__)


# ── Step 1: Merge transcript segments into a single string + build time index ──

def merge_segments(segments: list[dict]) -> tuple[str, list[dict]]:
    """
    Merges raw transcript segments into one continuous text string.
    Also builds a character-position → timestamp index so we can
    recover timestamps after chunking.

    Args:
        segments: [{"text": "...", "start": 14.32, "duration": 2.8}, ...]

    Returns:
        full_text: the entire transcript as one string
        time_index: [{"char_pos": 0, "start": 14.32, "end": 17.12}, ...]
    """
    full_text = ""
    time_index = []

    for seg in segments:
        char_pos = len(full_text)         # character position before appending
        text = seg["text"].strip()
        start = seg["start"]
        end = start + seg.get("duration", 0)

        time_index.append({
            "char_pos": char_pos,
            "start": start,
            "end": end,
        })

        full_text += text + " "           # space-join segments

    return full_text.strip(), time_index


# ── Step 2: Resolve timestamps for a chunk given its character span ───────────

def resolve_timestamps(
    chunk_start_char: int,
    chunk_end_char: int,
    time_index: list[dict],
) -> tuple[float, float]:
    """
    Given a chunk's character span in the full transcript, finds the
    corresponding start and end timestamps in seconds.

    Uses the time_index built in merge_segments to map char position → time.
    """
    chunk_start_time = 0.0
    chunk_end_time = 0.0

    for i, entry in enumerate(time_index):
        # Find the segment that contains the chunk's start character
        next_pos = time_index[i + 1]["char_pos"] if i + 1 < len(time_index) else float("inf")

        if entry["char_pos"] <= chunk_start_char < next_pos:
            chunk_start_time = entry["start"]

        if entry["char_pos"] <= chunk_end_char < next_pos:
            chunk_end_time = entry["end"]
            break

    return chunk_start_time, chunk_end_time


# ── Step 3: Chunk a single video's transcript ─────────────────────────────────

def chunk_video(transcript_path: Path) -> list[Document]:
    """
    Loads a transcript JSON file and returns a list of LangChain Documents.
    Each Document contains a text chunk + full metadata including timestamps.

    This is the core output unit — everything downstream works with Documents.
    """
    with open(transcript_path) as f:
        data = json.load(f)

    metadata = data["metadata"]
    segments = data["transcript_segments"]

    if not segments:
        logger.warning(f"Empty transcript: {transcript_path.stem}")
        return []

    # Merge all segments into one string + build time index
    full_text, time_index = merge_segments(segments)

    # Split into overlapping chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # prefer splitting at sentence boundaries
    )

    # split_text returns plain strings — we need char positions too
    # so we use create_documents which gives us offset tracking
    raw_chunks = splitter.create_documents([full_text])

    documents = []
    char_cursor = 0  # track position in full_text as we iterate chunks

    for chunk_doc in raw_chunks:
        chunk_text = chunk_doc.page_content

        # Skip noise chunks
        if len(chunk_text.strip()) < MIN_CHUNK_LENGTH:
            continue
        
        noise_tokens = ["[Music]", "[Applause]", "[Laughter]", "[music]"]
        noise_count = sum(chunk_text.count(token) for token in noise_tokens)
        word_count = len(chunk_text.split())
        if word_count > 0 and noise_count / word_count > 0.3:
            continue

        # Find where this chunk sits in the full_text
        chunk_start_char = full_text.find(chunk_text, char_cursor)
        if chunk_start_char == -1:
            chunk_start_char = char_cursor  # fallback
        chunk_end_char = chunk_start_char + len(chunk_text)

        # Resolve to timestamps
        t_start, t_end = resolve_timestamps(chunk_start_char, chunk_end_char, time_index)

        # Build the deep-link URL (timestamp in seconds)
        url_with_timestamp = (
            f"https://www.youtube.com/watch?v={metadata['video_id']}&t={int(t_start)}"
        )

        # Build LangChain Document with full metadata
        doc = Document(
            page_content=chunk_text,
            metadata={
                "video_id": metadata.get("video_id", ""),
                "video_title": metadata.get("video_title", ""),
                "channel_id": metadata.get("channel_id", ""),
                "published_date": metadata.get("published_date", ""),
                "duration_seconds": metadata.get("duration_seconds", 0),
                "chunk_start_time": t_start,
                "chunk_end_time": t_end,
                "url_with_timestamp": url_with_timestamp,
            },
        )
        documents.append(doc)
        char_cursor = chunk_start_char + 1  # advance cursor

    logger.info(f"Chunked '{metadata.get('video_title', '')}' → {len(documents)} chunks")
    return documents


# ── Step 4: Chunk entire channel ──────────────────────────────────────────────

def chunk_all_videos(transcripts_dir: Path = TRANSCRIPTS_DIR) -> list[Document]:
    """
    Loads and chunks every transcript JSON in the transcripts directory.
    Returns all Documents across all videos as a flat list.
    """
    transcript_files = sorted(transcripts_dir.glob("*.json"))

    if not transcript_files:
        raise FileNotFoundError(
            f"No transcript files found in {transcripts_dir}. Run ingest.py first."
        )

    all_documents = []
    for path in transcript_files:
        docs = chunk_video(path)
        all_documents.extend(docs)

    logger.info(f"Total chunks across all videos: {len(all_documents)}")
    return all_documents


# ── Entry point (for inspection/debugging) ────────────────────────────────────

if __name__ == "__main__":
    docs = chunk_all_videos()
    print(f"\nTotal documents: {len(docs)}")

    # Preview first chunk
    if docs:
        d = docs[0]
        print(f"\nSample chunk:")
        print(f"  Video : {d.metadata['video_title']}")
        print(f"  Time  : {d.metadata['chunk_start_time']:.0f}s → {d.metadata['chunk_end_time']:.0f}s")
        print(f"  URL   : {d.metadata['url_with_timestamp']}")
        print(f"  Text  : {d.page_content[:200]}...")