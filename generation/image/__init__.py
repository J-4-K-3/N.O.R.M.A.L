"""Image generation utilities for N.O.R.M.A.L.

This package provides a simple, extensible interface to generate images using
local backends (Diffusers) when installed, or remote APIs (Hugging Face / Replicate)
when tokens are provided. It also includes storage helpers.
"""

from .generator import generate_image
from .storage import save_image_bytes
from .utils import ensure_dir, safe_filename

__all__ = ["generate_image", "save_image_bytes", "ensure_dir", "safe_filename"]
