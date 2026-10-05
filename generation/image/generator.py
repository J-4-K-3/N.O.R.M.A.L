from __future__ import annotations

import base64
import io
import os
from typing import Optional, Dict

from PIL import Image
import requests

from app.generation.image.storage import save_image_bytes, save_pil_image

_LOCAL_AVAILABLE = False
try:
    import torch
    from diffusers import DiffusionPipeline
    _LOCAL_AVAILABLE = True
except Exception:
    _LOCAL_AVAILABLE = False


def _local_diffusers_generate(prompt: str, model: str, width: int, height: int, steps: int, guidance: float) -> bytes:
    pipe = DiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    images = pipe(prompt, height=height, width=width, num_inference_steps=steps, guidance_scale=guidance).images
    img = images[0]
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _hf_inference_generate(prompt: str, model: str, width: int, height: int, steps: int, guidance: float) -> bytes:
    token = os.getenv("HF_API_TOKEN")
    if not token:
        raise RuntimeError("HF_API_TOKEN not set for remote image generation")
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": prompt, "options": {"wait_for_model": True}}
    resp = requests.post(url, headers=headers, json=payload, stream=True, timeout=60)
    resp.raise_for_status()
    # If response is image bytes
    content_type = resp.headers.get("content-type", "")
    if content_type.startswith("image/"):
        return resp.content
    # Otherwise try parse JSON with base64
    try:
        data = resp.json()
        # common HF output may contain base64 in data[0]['generated_image'] or similar
        if isinstance(data, list) and data and isinstance(data[0], dict):
            for v in data[0].values():
                if isinstance(v, str) and v.startswith("data:image"):
                    b64 = v.split(",", 1)[1]
                    return base64.b64decode(b64)
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, str) and v.startswith("data:image"):
                    b64 = v.split(",", 1)[1]
                    return base64.b64decode(b64)
    except Exception:
        pass
    raise RuntimeError("Unexpected response from HF image inference API")


def generate_image(
    prompt: str,
    model: Optional[str] = None,
    width: int = 512,
    height: int = 512,
    steps: int = 30,
    guidance: float = 7.5,
    out_path: Optional[str] = None,
    prefer_local: bool = True,
) -> Dict[str, str]:
    """Generate an image from `prompt` and save it to disk.

    Returns a dict: {"path": str, "error": str}
    """
    model = model or os.getenv("AI_IMAGE_MODEL", "runwayml/stable-diffusion-v1-5")

    # Try local generation
    if prefer_local and _LOCAL_AVAILABLE:
        try:
            data = _local_diffusers_generate(prompt, model, width, height, steps, guidance)
            filename = out_path or None
            if filename and os.path.isabs(filename):
                # write directly
                path = save_image_bytes(data, filename=os.path.basename(filename), out_dir=os.path.dirname(filename))
            else:
                path = save_image_bytes(data, filename=None)
            return {"path": path}
        except Exception as exc:
            # fallthrough to remote
            local_err = str(exc)
    else:
        local_err = "local backend not available"

    # Remote HF inference fallback
    try:
        data = _hf_inference_generate(prompt, model, width, height, steps, guidance)
        path = save_image_bytes(data)
        return {"path": path}
    except Exception as exc:
        return {"error": f"image generation failed: {exc} (local_err: {local_err})"}
