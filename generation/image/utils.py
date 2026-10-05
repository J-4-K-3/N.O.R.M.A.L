from __future__ import annotations

import os
import re
from typing import Optional


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_filename(text: str, ext: str = "png") -> str:
    name = re.sub(r"[^a-zA-Z0-9_-]", "-", text).strip("-_")
    if not name:
        name = "image"
    return f"{name}.{ext}"
