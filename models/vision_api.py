"""Vision API adapters.

Wraps local vision models (CLIP, ViT) or external APIs for image understanding,
captioning, and object detection. Keep implementations lazy to avoid heavy
imports unless used.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional


class VisionAPI:
    def __init__(self, backend: Optional[str] = None):
        self.backend = backend or os.getenv("AI_VISION_BACKEND", "local_clip")

    def describe_image(self, image_bytes: bytes) -> Dict[str, Any]:
        # local CLIP-like description (placeholder)
        try:
            from PIL import Image
            # real CLIP integration would go here
        except Exception:
            pass
        # remote fallback
        safety_api = os.getenv("VISION_API_URL")
        if safety_api:
            import requests
            resp = requests.post(safety_api, json={"image": image_bytes.decode("latin1")}, timeout=15)
            try:
                return resp.json()
            except Exception:
                return {"text": resp.text}

        return {"caption": "[vision caption placeholder]"}


def get_vision_api() -> VisionAPI:
    return VisionAPI()
