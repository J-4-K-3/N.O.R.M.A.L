from __future__ import annotations

import base64
import os
import time
from typing import Optional

from PIL import Image
from io import BytesIO

from app.generation.image.utils import ensure_dir, safe_filename


DEFAULT_DIR = os.path.join("data", "generated_images")


def save_image_bytes(data: bytes, filename: Optional[str] = None, out_dir: Optional[str] = None) -> str:
    out_dir = out_dir or DEFAULT_DIR
    ensure_dir(out_dir)
    if not filename:
        filename = f"image_{int(time.time())}.png"
    path = os.path.join(out_dir, filename)
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def save_base64_image(b64string: str, filename: Optional[str] = None, out_dir: Optional[str] = None) -> str:
    data = base64.b64decode(b64string)
    return save_image_bytes(data, filename=filename, out_dir=out_dir)


def save_pil_image(img: Image.Image, filename: Optional[str] = None, out_dir: Optional[str] = None) -> str:
    out_dir = out_dir or DEFAULT_DIR
    ensure_dir(out_dir)
    if not filename:
        filename = safe_filename("image")
    path = os.path.join(out_dir, filename)
    img.save(path)
    return path
