# src/ingest.py
"""
Fetches all video transcripts + metadata from a YouTube channel.

Pipeline:
  Channel ID → list of video IDs → per-video transcript + metadata → saved as JSON

Output: one JSON file per video in data/transcripts/
Each file contains: video metadata + list of transcript segments with timestamps.
"""

import json
import time
import logging
from pathlib import Path
from typing import Optional

import scrapetube
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from youtube_transcript_api.formatters import JSONFormatter

from config import (
    CHANNEL_ID,
    TRANSCRIPTS_DIR,
    MAX_VIDEOS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


# ── Fetch video list ──────────────────────────────────────────────────────────

def get_channel_video_ids(channel_id: str, max_videos: Optional[int] = None) -> list[str]:
    """
    Returns a list of video IDs for every public video on the channel.
    scrapetube scrapes the channel page without requiring a YouTube Data API key.
    """
    logger.info(f"Fetching video list for channel: {channel_id}")
    videos = scrapetube.get_channel(channel_id)

    video_ids = []
    for video in videos:
        video_ids.append(video["videoId"])
        if max_videos and len(video_ids) >= max_videos:
            break

    logger.info(f"Found {len(video_ids)} videos")
    return video_ids


# ── Fetch transcript for a single video ───────────────────────────────────────

def fetch_transcript(video_id: str) -> Optional[list[dict]]:
    """
    Returns transcript as a list of segments:
    [{"text": "...", "start": 14.32, "duration": 3.1}, ...]

    start = seconds from beginning of video (this becomes chunk_start_time later)
    Returns None if transcript unavailable (private/disabled).
    """
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        return transcript
    except TranscriptsDisabled:
        logger.warning(f"Transcripts disabled for video: {video_id}")
        return None
    except NoTranscriptFound:
        logger.warning(f"No English transcript for video: {video_id}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error for video {video_id}: {e}")
        return None


# ── Fetch video metadata via yt-dlp ───────────────────────────────────────────

def fetch_video_metadata(video_id: str) -> dict:
    """
    Fetches title, published_date, duration using yt-dlp.
    yt-dlp is used only for metadata (not downloading video) — fast and no API key needed.
    """
    import yt_dlp

    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        "quiet": True,
        "skip_download": True,           # metadata only, no video download
        "extract_flat": False,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                "video_id": video_id,
                "video_title": info.get("title", ""),
                "channel_id": info.get("channel_id", ""),
                "published_date": info.get("upload_date", ""),  # "YYYYMMDD"
                "duration_seconds": info.get("duration", 0),
                "view_count": info.get("view_count", 0),
                "url": f"https://www.youtube.com/watch?v={video_id}",
            }
    except Exception as e:
        logger.error(f"Failed to fetch metadata for {video_id}: {e}")
        return {"video_id": video_id}


# ── Combine transcript + metadata → save to disk ──────────────────────────────

def process_video(video_id: str, output_dir: Path) -> bool:
    """
    Fetches transcript + metadata for a single video and saves as JSON.
    Returns True if successful, False if skipped.
    """
    output_path = output_dir / f"{video_id}.json"

    # Skip if already processed (allows resuming interrupted ingestion)
    if output_path.exists():
        logger.info(f"Already processed: {video_id} — skipping")
        return True

    transcript = fetch_transcript(video_id)
    if transcript is None:
        return False  # No transcript available — skip this video

    metadata = fetch_video_metadata(video_id)

    # Combine into a single document object
    document = {
        "metadata": metadata,
        "transcript_segments": transcript,
        # transcript_segments shape:
        # [{"text": "...", "start": 14.32, "duration": 3.1}, ...]
    }

    output_path.write_text(json.dumps(document, indent=2, ensure_ascii=False))
    logger.info(f"Saved: {metadata.get('video_title', video_id)}")
    return True


# ── Main ingestion loop ────────────────────────────────────────────────────────

def ingest_channel(
    channel_id: str = CHANNEL_ID,
    output_dir: Path = TRANSCRIPTS_DIR,
    max_videos: Optional[int] = MAX_VIDEOS,
    delay_seconds: float = 1.0,          # polite crawl delay between requests
) -> dict:
    """
    Full pipeline: channel ID → transcript JSON files on disk.

    Args:
        delay_seconds: sleep between videos to avoid rate limiting.

    Returns:
        Summary dict with counts of processed/skipped/failed videos.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    video_ids = get_channel_video_ids(channel_id, max_videos)

    stats = {"total": len(video_ids), "processed": 0, "skipped": 0, "failed": 0}

    for i, video_id in enumerate(video_ids):
        logger.info(f"[{i+1}/{stats['total']}] Processing: {video_id}")

        success = process_video(video_id, output_dir)

        if success:
            stats["processed"] += 1
        else:
            stats["skipped"] += 1

        time.sleep(delay_seconds)  # be polite to YouTube's servers

    logger.info(f"Ingestion complete: {stats}")
    return stats


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    stats = ingest_channel()
    print(f"\n✅ Done. Processed: {stats['processed']} | Skipped: {stats['skipped']}")