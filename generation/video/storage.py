from __future__ import annotations

import os
import time
from typing import Optional

from app.generation.video.utils import ensure_dir, safe_filename


DEFAULT_DIR = os.path.join("data", "generated_videos")


def save_video_file(data_path: str, filename: Optional[str] = None, out_dir: Optional[str] = None) -> str:
    out_dir = out_dir or DEFAULT_DIR
    ensure_dir(out_dir)
    if not filename:
        filename = safe_filename("video")
    dest = os.path.join(out_dir, filename)
    # copy file
    try:
        with open(data_path, "rb") as rf, open(dest, "wb") as wf:
            wf.write(rf.read())
    except Exception:
        # fallback: attempt to move
        try:
            os.replace(data_path, dest)
        except Exception:
            raise
    return dest
