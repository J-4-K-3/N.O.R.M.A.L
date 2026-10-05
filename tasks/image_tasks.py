from __future__ import annotations

from typing import Optional
import os

from .celery_app import celery_app

from app.generation.image.generator import generate_image
from app.generation.image.upscale import upscale_image_bytes
from app.generation.image.s3 import upload_bytes_to_s3
from app.generation.image.safety import check_image_safety_bytes, check_prompt_safety
from app.tools.logger import log_generation_event


@celery_app.task(bind=True)
def generate_image_job(
    self,
    prompt: str,
    model: Optional[str] = None,
    width: int = 512,
    height: int = 512,
    steps: int = 30,
    guidance: float = 7.5,
    upscale: bool = False,
    upscale_scale: int = 2,
    upload_s3: bool = False,
    user: Optional[str] = None,
):
    """Celery task: generate image, optionally upscale, run safety, and upload to S3."""
    prompt_guard = check_prompt_safety(prompt)
    if not prompt_guard.get("safe", True):
        payload = {
            "status": "blocked",
            "reason": "prompt_policy_block",
            "safety": prompt_guard,
        }
        if user is not None:
            payload["user"] = user
        log_generation_event(task_id=getattr(self, "request", None).id if getattr(self, "request", None) else None, status="blocked", prompt=prompt, safety=prompt_guard)
        return payload

    log_generation_event(task_id=getattr(self, "request", None).id if getattr(self, "request", None) else None, status="queued", prompt=prompt, model=model)

    gen = generate_image(prompt, model=model, width=width, height=height, steps=steps, guidance=guidance)
    if "error" in gen:
        log_generation_event(task_id=getattr(self, "request", None).id if getattr(self, "request", None) else None, status="failed", prompt=prompt, error=gen["error"])
        return {"status": "failed", "error": gen["error"], **({"user": user} if user is not None else {})}

    path = gen.get("path")
    if not path:
        log_generation_event(task_id=getattr(self, "request", None).id if getattr(self, "request", None) else None, status="failed", prompt=prompt, error="no output path")
        return {"status": "failed", "error": "no output path", **({"user": user} if user is not None else {})}

    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except Exception as e:
        log_generation_event(task_id=getattr(self, "request", None).id if getattr(self, "request", None) else None, status="failed", prompt=prompt, error=f"read error: {e}")
        return {"status": "failed", "error": f"read error: {e}", **({"user": user} if user is not None else {})}

    result = {"local_path": path}
    if user is not None:
        result["user"] = user

    if upscale:
        try:
            data = upscale_image_bytes(data, scale=upscale_scale)
            with open(path, "wb") as fh:
                fh.write(data)
            result["upscaled"] = True
        except Exception as e:
            result["upscale_error"] = str(e)

    safety = check_image_safety_bytes(data, prompt=prompt)
    result["safety"] = safety
    if not safety.get("safe", True):
        result["status"] = "blocked"
        log_generation_event(task_id=getattr(self, "request", None).id if getattr(self, "request", None) else None, status="blocked", prompt=prompt, safety=safety)
        return result

    if upload_s3 and os.getenv("S3_BUCKET"):
        try:
            bucket = os.getenv("S3_BUCKET")
            key = f"images/{os.path.basename(path)}"
            s3url = upload_bytes_to_s3(data, bucket, key)
            result["s3_url"] = s3url
        except Exception as e:
            result["s3_error"] = str(e)

    result["status"] = "completed"
    log_generation_event(task_id=getattr(self, "request", None).id if getattr(self, "request", None) else None, status="completed", prompt=prompt, local_path=path, safety=safety)
    return result
