"""Video generation utilities for N.O.R.M.A.L.

Provides a simple pipeline that generates a sequence of frames (via the
image generator) and stitches them into a video. This is a scaffold — replace
frame generation with advanced video models for production quality.
"""

from .generator import generate_video
from .storage import save_video_file
from .utils import ensure_dir, safe_filename

__all__ = ["generate_video", "save_video_file", "ensure_dir", "safe_filename"]
