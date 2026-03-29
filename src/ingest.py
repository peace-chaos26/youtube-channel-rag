# src/ingest.py

import json
import re
import time
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

import scrapetube
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

from config import get_transcripts_dir, VIDEO_COUNT_WARNING_THRESHOLD

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


# ── Resolve any YouTube channel URL to a channel ID ──────────────────────────

def resolve_channel_id(channel_url: str) -> Optional[str]:
    """
    Accepts any of these formats and returns the channel ID:
      https://www.youtube.com/@AndrejKarpathy
      https://www.youtube.com/channel/UCXUPKJO5MZQN11PqgIvyuvQ
      UCXUPKJO5MZQN11PqgIvyuvQ   (raw ID)

    Uses yt-dlp to resolve handle → channel ID reliably.
    """
    import yt_dlp

    # Already a raw channel ID (starts with UC, 24 chars)
    if re.match(r'^UC[\w-]{22}$', channel_url.strip()):
        return channel_url.strip()

    # Normalise URL
    url = channel_url.strip()
    if not url.startswith("http"):
        url = f"https://www.youtube.com/{url}"

    ydl_opts = {"quiet": True, "skip_download": True, "extract_flat": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            channel_id = info.get("channel_id") or info.get("uploader_id")
            if channel_id:
                logger.info(f"Resolved channel ID: {channel_id}")
                return channel_id
    except Exception as e:
        logger.error(f"Failed to resolve channel ID from {url}: {e}")

    return None


# ── Get video count without full ingestion ────────────────────────────────────

def get_video_count(channel_id: str) -> int:
    """Quick count of public videos on a channel — used for the warning threshold."""
    try:
        videos = list(scrapetube.get_channel(channel_id))
        return len(videos)
    except Exception as e:
        logger.error(f"Failed to count videos: {e}")
        return 0


# ── Get video IDs ─────────────────────────────────────────────────────────────

def get_channel_video_ids(channel_id: str, max_videos: Optional[int] = None) -> list[str]:
    logger.info(f"Fetching video list for channel: {channel_id}")
    videos = scrapetube.get_channel(channel_id)

    video_ids = []
    for video in videos:
        video_ids.append(video["videoId"])
        if max_videos and len(video_ids) >= max_videos:
            break

    logger.info(f"Found {len(video_ids)} videos")
    return video_ids


# ── Fetch transcript ──────────────────────────────────────────────────────────

def fetch_transcript(video_id: str) -> Optional[list[dict]]:
    try:
        ytt_api = YouTubeTranscriptApi()
        fetched = ytt_api.fetch(video_id)
        return [
            {"text": snippet.text, "start": snippet.start, "duration": snippet.duration}
            for snippet in fetched
        ]
    except Exception as e:
        logger.warning(f"No transcript for video {video_id}: {e}")
        return None


# ── Fetch metadata ────────────────────────────────────────────────────────────

def fetch_video_metadata(video_id: str) -> dict:
    import yt_dlp
    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {"quiet": True, "skip_download": True, "extract_flat": False}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                "video_id": video_id,
                "video_title": info.get("title", ""),
                "channel_id": info.get("channel_id", ""),
                "channel_name": info.get("uploader", ""),
                "published_date": info.get("upload_date", ""),
                "duration_seconds": info.get("duration", 0),
                "url": f"https://www.youtube.com/watch?v={video_id}",
            }
    except Exception as e:
        logger.error(f"Failed to fetch metadata for {video_id}: {e}")
        return {"video_id": video_id}


# ── Process single video ──────────────────────────────────────────────────────

def process_video(video_id: str, output_dir: Path) -> bool:
    output_path = output_dir / f"{video_id}.json"
    if output_path.exists():
        logger.info(f"Already processed: {video_id} — skipping")
        return True

    transcript = fetch_transcript(video_id)
    if transcript is None:
        return False

    metadata = fetch_video_metadata(video_id)
    document = {"metadata": metadata, "transcript_segments": transcript}
    output_path.write_text(json.dumps(document, indent=2, ensure_ascii=False))
    logger.info(f"Saved: {metadata.get('video_title', video_id)}")
    return True


# ── Main ingestion ────────────────────────────────────────────────────────────

def ingest_channel(
    channel_id: str,
    max_videos: Optional[int] = None,
    delay_seconds: float = 1.0,
    progress_callback=None,       # optional callable(current, total) for UI progress
) -> dict:
    """
    Full ingestion pipeline for a channel.
    progress_callback lets the Streamlit UI update a progress bar in real time.
    """
    output_dir = get_transcripts_dir(channel_id)
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

        if progress_callback:
            progress_callback(i + 1, stats["total"])

        time.sleep(delay_seconds)

    logger.info(f"Ingestion complete: {stats}")
    return stats