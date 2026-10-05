from __future__ import annotations

from typing import Optional
import os

from .celery_app import celery_app

from app.generation.image.safety import check_image_safety_bytes, check_prompt_safety, check_copyright_risk
from app.generation.image.s3 import upload_bytes_to_s3
from app.generation.video.generator import generate_music, generate_video
from app.tools.logger import log_generation_event


@celery_app.task(bind=True)
def generate_music_job(
    self,
    prompt: str,
    duration: float = 8.0,
    sample_rate: int = 22050,
    model: Optional[str] = None,
    upload_s3: bool = False,
    allow_copyrighted: bool = False,
    user: Optional[str] = None,
):
    """Generate a short symbolic audio clip from a prompt and record status metadata."""
    prompt_guard = check_prompt_safety(prompt)
    copyright_guard = check_copyright_risk(prompt, allow_copyrighted=allow_copyrighted)
    if not prompt_guard.get("safe", True):
        payload = {"status": "blocked", "reason": "prompt_policy_block", "safety": prompt_guard}
        if user is not None:
            payload["user"] = user
        log_generation_event(task_id=getattr(self, "request", None).id if getattr(self, "request", None) else None, status="blocked", prompt=prompt, safety=prompt_guard)
        return payload
    if not copyright_guard.get("safe", True):
        payload = {"status": "blocked", "reason": "copyright_policy_block", "copyright": copyright_guard}
        if user is not None:
            payload["user"] = user
        log_generation_event(task_id=getattr(self, "request", None).id if getattr(self, "request", None) else None, status="blocked", prompt=prompt, copyright=copyright_guard)
        return payload

    result = generate_music(prompt, duration=duration, sample_rate=sample_rate, model=model)
    if result.get("error"):
        log_generation_event(task_id=getattr(self, "request", None).id if getattr(self, "request", None) else None, status="failed", prompt=prompt, error=result["error"])
        return {"status": "failed", "error": result["error"], **({"user": user} if user is not None else {})}

    music_path = result.get("music_path")
    response = {"status": "completed", "music_path": music_path}
    if user is not None:
        response["user"] = user
    if upload_s3 and os.getenv("S3_BUCKET"):
        try:
            bucket = os.getenv("S3_BUCKET")
            key = f"music/{os.path.basename(music_path)}"
            with open(music_path, "rb") as fh:
                payload = fh.read()
            s3url = upload_bytes_to_s3(payload, bucket, key, content_type="audio/wav")
            response["s3_url"] = s3url
        except Exception as exc:
            response["s3_error"] = str(exc)

    log_generation_event(task_id=getattr(self, "request", None).id if getattr(self, "request", None) else None, status="completed", prompt=prompt, music_path=music_path)
    return response


@celery_app.task(bind=True)
def generate_video_job(
    self,
    prompt: str,
    frames: int = 8,
    fps: int = 8,
    model: Optional[str] = None,
    width: int = 512,
    height: int = 512,
    steps: int = 30,
    guidance: float = 7.5,
    upload_s3: bool = False,
    temporal_consistency: bool = True,
    motion_prior: Optional[str] = None,
    cache_frames: bool = True,
    allow_copyrighted: bool = False,
    user: Optional[str] = None,
):
    """Generate a short low-resolution video using a frame-based diffusion pipeline with temporal stabilization and caching."""
    prompt_guard = check_prompt_safety(prompt)
    copyright_guard = check_copyright_risk(prompt, allow_copyrighted=allow_copyrighted)
    if not prompt_guard.get("safe", True):
        payload = {"status": "blocked", "reason": "prompt_policy_block", "safety": prompt_guard}
        if user is not None:
            payload["user"] = user
        log_generation_event(task_id=getattr(self, "request", None).id if getattr(self, "request", None) else None, status="blocked", prompt=prompt, safety=prompt_guard)
        return payload
    if not copyright_guard.get("safe", True):
        payload = {"status": "blocked", "reason": "copyright_policy_block", "copyright": copyright_guard}
        if user is not None:
            payload["user"] = user
        log_generation_event(task_id=getattr(self, "request", None).id if getattr(self, "request", None) else None, status="blocked", prompt=prompt, copyright=copyright_guard)
        return payload

    res = generate_video(
        prompt,
        frames=frames,
        fps=fps,
        model=model,
        width=width,
        height=height,
        steps=steps,
        guidance=guidance,
        temporal_consistency=temporal_consistency,
        motion_prior=motion_prior,
        cache_frames=cache_frames,
    )
    if res.get("video_path") is None:
        log_generation_event(task_id=getattr(self, "request", None).id if getattr(self, "request", None) else None, status="failed", prompt=prompt, error=res.get("error"))
        return {"status": "failed", "error": res.get("error"), **({"user": user} if user is not None else {})}

    video_path = res.get("video_path")
    frames_list = res.get("frames", [])

    result = {"video_path": video_path, "motion_prior": res.get("motion_prior"), "temporal_consistency": res.get("temporal_consistency")}
    if user is not None:
        result["user"] = user

    if frames_list:
        try:
            with open(frames_list[0], "rb") as fh:
                data = fh.read()
            safety = check_image_safety_bytes(data, prompt=prompt)
            result["safety"] = safety
            if not safety.get("safe", True):
                result["status"] = "blocked"
                log_generation_event(task_id=getattr(self, "request", None).id if getattr(self, "request", None) else None, status="blocked", prompt=prompt, safety=safety)
                return result
        except Exception as e:
            result["safety_error"] = str(e)

    if upload_s3 and os.getenv("S3_BUCKET"):
        try:
            bucket = os.getenv("S3_BUCKET")
            key = f"videos/{os.path.basename(video_path)}"
            with open(video_path, "rb") as fh:
                vbytes = fh.read()
            s3url = upload_bytes_to_s3(vbytes, bucket, key, content_type="video/mp4")
            result["s3_url"] = s3url
        except Exception as e:
            result["s3_error"] = str(e)

    result["status"] = "completed"
    log_generation_event(task_id=getattr(self, "request", None).id if getattr(self, "request", None) else None, status="completed", prompt=prompt, video_path=video_path)
    return result
