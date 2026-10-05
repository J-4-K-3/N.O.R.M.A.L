from __future__ import annotations

import io
from typing import Optional

from PIL import Image


def upscale_image_bytes(data: bytes, scale: int = 2, resample: Optional[str] = None) -> bytes:
    """Upscale image bytes using Pillow. Returns new image bytes (PNG).

    This is a simple, fast upscaler. For production quality upscaling, replace
    with Real-ESRGAN or an optimized model.
    """
    img = Image.open(io.BytesIO(data)).convert("RGBA")
    w, h = img.size
    new_size = (int(w * scale), int(h * scale))
    # Use high-quality resampling
    resample_filter = Image.LANCZOS
    up = img.resize(new_size, resample=resample_filter)
    buf = io.BytesIO()
    up.save(buf, format="PNG")
    return buf.getvalue()
