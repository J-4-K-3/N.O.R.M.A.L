import os
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.core.auth import require_auth
from app.core.config import (
    get_data_engineering_config,
    get_ethics_config,
    get_generation_presets,
    get_mlop_config,
    list_model_registry,
    resolve_model_variant,
)
from app.core.ratelimit import check_rate_limit
from app.generation.image.generator import generate_image
from app.generation.indexer import SimpleIndexer
from app.generation.nodes import NodeManager
from app.generation.video.generator import generate_music, generate_video
from app.tasks.generation_tasks import generate_job
from app.tasks.image_tasks import generate_image_job
from app.tasks.video_tasks import generate_music_job, generate_video_job
from app.tools.logger import log_feedback_event, read_recent_events, summarize_events

router = APIRouter()


def _track_task_owner(task_id: str, user: str) -> None:
    """Persist a best-effort owner mapping for the task ID to guard status endpoints."""
    try:
        import redis

        redis_url = os.getenv("REDIS_URL")
        if not redis_url or redis is None:
            return
        client = redis.from_url(redis_url, decode_responses=True)
        client.hset(
            f"task:{task_id}:meta",
            mapping={"owner": user, "created_at": str(int(time.time()))},
        )
        client.expire(f"task:{task_id}:meta", 86400)
    except Exception:
        # Best effort only; do not break generation for missing Redis metadata.
        pass


def _get_task_owner(task_id: str) -> Optional[str]:
    try:
        import redis

        redis_url = os.getenv("REDIS_URL")
        if not redis_url or redis is None:
            return None
        client = redis.from_url(redis_url, decode_responses=True)
        return client.hget(f"task:{task_id}:meta", "owner")
    except Exception:
        return None


def _task_result_for_user(task_id: str, user: str) -> Dict[str, Any]:
    from app.tasks.celery_app import celery_app

    res = celery_app.AsyncResult(task_id)
    result = res.result

    redis_owner = _get_task_owner(task_id)
    if redis_owner and redis_owner != user:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="task is not authorized for this user")

    if isinstance(result, dict):
        owner = result.get("user")
        if owner is not None and owner != user:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="task is not authorized for this user")

    if result is None and res.state in {"PENDING", "STARTED", "RETRY"}:
        return {"task_id": task_id, "status": res.status, "result": {"status": res.state.lower()}}

    return {"task_id": task_id, "status": res.status, "result": result}


def _resolve_generation_model(payload_model: Optional[str], user: Optional[str] = None) -> str:
    if payload_model:
        return payload_model
    return resolve_model_variant("image", user_id=user, request_id=None)


class DemoRequest(BaseModel):
    input_text: str
    index_name: Optional[str] = "default"
    index_docs: Optional[List[Dict[str, Any]]] = None


class DemoResponse(BaseModel):
    output: str
    debug: Dict[str, Any] = Field(default_factory=dict)


@router.post("/generation/demo", response_model=DemoResponse)
def generation_demo(payload: DemoRequest, user: str = Depends(require_auth)):
    """Synchronous demo path intended for controlled internal/demo usage."""
    check_rate_limit(user)
    indexer = SimpleIndexer(name=payload.index_name)
    if payload.index_docs:
        indexer.index_documents(payload.index_docs)

    node_manager = NodeManager(indexer=indexer, connectors=[])
    result = node_manager.run_pipeline(payload.input_text)
    return {"output": result.get("output", ""), "debug": result.get("debug", {})}


class IndexRequest(BaseModel):
    index_name: str = "default"
    documents: List[Dict[str, Any]]


@router.post("/generation/index")
def index_documents(payload: IndexRequest, user: str = Depends(require_auth)):
    check_rate_limit(user)
    indexer = SimpleIndexer(name=payload.index_name)
    indexer.index_documents(payload.documents)
    return {"status": "ok", "indexed": len(payload.documents)}


class QueueRequest(BaseModel):
    input_text: str
    index_name: Optional[str] = "default"


@router.post("/generation/queue")
def queue_generation(payload: QueueRequest, user: str = Depends(require_auth)):
    check_rate_limit(user)
    task = generate_job.delay(payload.input_text, payload.index_name, user=user)
    _track_task_owner(task.id, user)
    return {"task_id": task.id}


@router.get("/generation/status/{task_id}")
def generation_status(task_id: str, user: str = Depends(require_auth)):
    return _task_result_for_user(task_id, user)


class ImageDemoRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    width: int = Field(default=512, ge=64, le=2048)
    height: int = Field(default=512, ge=64, le=2048)
    steps: int = Field(default=30, ge=1, le=200)
    guidance: float = Field(default=7.5, ge=0.0, le=30.0)


class ImageQueueRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    width: int = Field(default=512, ge=64, le=2048)
    height: int = Field(default=512, ge=64, le=2048)
    steps: int = Field(default=30, ge=1, le=200)
    guidance: float = Field(default=7.5, ge=0.0, le=30.0)
    upscale: bool = False
    upscale_scale: int = Field(default=2, ge=1, le=8)
    upload_s3: bool = False


@router.post("/generation/image/demo")
def image_demo(payload: ImageDemoRequest, user: str = Depends(require_auth)):
    check_rate_limit(user)
    model_name = _resolve_generation_model(payload.model, user=user)
    res = generate_image(
        payload.prompt,
        model=model_name,
        width=payload.width,
        height=payload.height,
        steps=payload.steps,
        guidance=payload.guidance,
    )
    return {"result": res, "model": model_name}


@router.post("/generation/image/queue")
def image_queue(payload: ImageQueueRequest, user: str = Depends(require_auth)):
    check_rate_limit(user)
    model_name = _resolve_generation_model(payload.model, user=user)
    task = generate_image_job.delay(
        payload.prompt,
        model_name,
        payload.width,
        payload.height,
        payload.steps,
        payload.guidance,
        payload.upscale,
        payload.upscale_scale,
        payload.upload_s3,
        user=user,
    )
    _track_task_owner(task.id, user)
    return {"task_id": task.id, "model": model_name}


@router.get("/generation/image/status/{task_id}")
def image_status(task_id: str, user: str = Depends(require_auth)):
    return _task_result_for_user(task_id, user)


class VideoDemoRequest(BaseModel):
    prompt: str
    frames: int = Field(default=8, ge=1, le=120)
    fps: int = Field(default=8, ge=1, le=30)
    model: Optional[str] = None
    width: int = Field(default=512, ge=64, le=2048)
    height: int = Field(default=512, ge=64, le=2048)
    steps: int = Field(default=30, ge=1, le=200)
    guidance: float = Field(default=7.5, ge=0.0, le=30.0)
    temporal_consistency: bool = True
    motion_prior: Optional[str] = None
    cache_frames: bool = True
    allow_copyrighted: bool = False


class VideoQueueRequest(BaseModel):
    prompt: str
    frames: int = Field(default=8, ge=1, le=120)
    fps: int = Field(default=8, ge=1, le=30)
    model: Optional[str] = None
    width: int = Field(default=512, ge=64, le=2048)
    height: int = Field(default=512, ge=64, le=2048)
    steps: int = Field(default=30, ge=1, le=200)
    guidance: float = Field(default=7.5, ge=0.0, le=30.0)
    upload_s3: bool = False
    temporal_consistency: bool = True
    motion_prior: Optional[str] = None
    cache_frames: bool = True
    allow_copyrighted: bool = False


@router.post("/generation/video/demo")
def video_demo(payload: VideoDemoRequest, user: str = Depends(require_auth)):
    check_rate_limit(user)
    res = generate_video(
        payload.prompt,
        frames=payload.frames,
        fps=payload.fps,
        model=payload.model,
        width=payload.width,
        height=payload.height,
        steps=payload.steps,
        guidance=payload.guidance,
        temporal_consistency=payload.temporal_consistency,
        motion_prior=payload.motion_prior,
        cache_frames=payload.cache_frames,
        allow_copyrighted=payload.allow_copyrighted,
    )
    return {"result": res}


@router.post("/generation/video/queue")
def video_queue(payload: VideoQueueRequest, user: str = Depends(require_auth)):
    check_rate_limit(user)
    task = generate_video_job.delay(
        payload.prompt,
        payload.frames,
        payload.fps,
        payload.model,
        payload.width,
        payload.height,
        payload.steps,
        payload.guidance,
        payload.upload_s3,
        payload.temporal_consistency,
        payload.motion_prior,
        payload.cache_frames,
        allow_copyrighted=payload.allow_copyrighted,
        user=user,
    )
    _track_task_owner(task.id, user)
    return {"task_id": task.id}


@router.get("/generation/video/status/{task_id}")
def video_status(task_id: str, user: str = Depends(require_auth)):
    return _task_result_for_user(task_id, user)


class MusicDemoRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    duration: float = Field(default=8.0, ge=1.0, le=180.0)
    sample_rate: int = Field(default=22050, ge=8000, le=48000)
    allow_copyrighted: bool = False


class MusicQueueRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    duration: float = Field(default=8.0, ge=1.0, le=180.0)
    sample_rate: int = Field(default=22050, ge=8000, le=48000)
    upload_s3: bool = False
    allow_copyrighted: bool = False


@router.post("/generation/music/demo")
def music_demo(payload: MusicDemoRequest, user: str = Depends(require_auth)):
    check_rate_limit(user)
    res = generate_music(
        payload.prompt,
        duration=payload.duration,
        sample_rate=payload.sample_rate,
        model=payload.model,
        allow_copyrighted=payload.allow_copyrighted,
    )
    return {"result": res}


@router.post("/generation/music/queue")
def music_queue(payload: MusicQueueRequest, user: str = Depends(require_auth)):
    check_rate_limit(user)
    task = generate_music_job.delay(
        payload.prompt,
        payload.duration,
        payload.sample_rate,
        payload.model,
        payload.upload_s3,
        allow_copyrighted=payload.allow_copyrighted,
        user=user,
    )
    _track_task_owner(task.id, user)
    return {"task_id": task.id}


@router.get("/generation/music/status/{task_id}")
def music_status(task_id: str, user: str = Depends(require_auth)):
    return _task_result_for_user(task_id, user)


@router.get("/generation/analytics")
def generation_analytics(limit: int = 25):
    events = read_recent_events(limit=limit)
    return {"status": "ok", "summary": summarize_events(events)}


@router.get("/generation/models")
def generation_models(scope: str = "image"):
    return {"status": "ok", "models": list_model_registry(scope=scope)}


@router.get("/generation/presets")
def generation_presets():
    return {"status": "ok", "presets": get_generation_presets()}


class FeedbackRequest(BaseModel):
    prompt: Optional[str] = None
    model: Optional[str] = None
    output_path: Optional[str] = None
    rating: Optional[int] = Field(default=None, ge=1, le=5)
    notes: Optional[str] = None
    feedback_type: Optional[str] = "quality"


@router.post("/generation/feedback")
def generation_feedback(payload: FeedbackRequest, user: str = Depends(require_auth)):
    check_rate_limit(user)
    event = log_feedback_event(
        payload.feedback_type or "quality",
        user=user,
        prompt=payload.prompt,
        model=payload.model,
        output_path=payload.output_path,
        rating=payload.rating,
        notes=payload.notes,
    )
    return {"status": "ok", "event": event}


@router.get("/generation/platform")
def generation_platform():
    return {
        "status": "ok",
        "data_engineering": get_data_engineering_config(),
        "mlops": get_mlop_config(),
        "ethics": get_ethics_config(),
    }


