"""Locally stored reference videos for native signs, so they play with no internet connection.

Drop an MP4 into `dataset/videos/` (git-ignored, never deployed) named after the sign: `hello.mp4`, `book.mp4`, ... A few
glosses need an explicit file name (see FILE_NAME_OVERRIDES). Only files that exist and sit directly inside that folder are
ever served, and only for a known gloss, so the endpoint can never read an arbitrary path.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
LOCAL_VIDEO_DIR = Path(os.getenv("ASL_QUEST_LOCAL_VIDEO_DIR", "").strip() or ROOT / "dataset" / "videos")

# gloss -> file name, where the default (lower-cased gloss + ".mp4") is not the file's name.
FILE_NAME_OVERRIDES = {
    "THANKYOU": "thank-you.mp4",
    "EAT1": "eat.mp4",
}

_GLOSS_RE = re.compile(r"^[A-Z0-9_]{1,40}$")


def local_video_path(gloss: str, directory: Path | None = None) -> Path | None:
    """The local MP4 for `gloss`, or None when there is none (or the gloss is not a plain catalog-style name)."""
    gloss = (gloss or "").strip().upper()
    if not _GLOSS_RE.match(gloss):
        return None
    base = (directory or LOCAL_VIDEO_DIR).resolve()
    candidate = (base / FILE_NAME_OVERRIDES.get(gloss, f"{gloss.lower()}.mp4")).resolve()
    if candidate.parent != base or not candidate.is_file():
        return None
    return candidate


def local_video_url(gloss: str) -> str | None:
    """Path (relative to the API root) the frontend can put in a <video src>, when a local file exists."""
    return f"/native/reference-video/{gloss.strip().upper()}" if local_video_path(gloss) else None
