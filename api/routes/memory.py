# /memory endpoints

from fastapi import APIRouter
from pydantic import BaseModel

from app.services.memory import save_memory, get_recent_memories

router = APIRouter()


class MemorySaveRequest(BaseModel):
    text: str


@router.post("/memory/save")
def memory_save(payload: MemorySaveRequest):
    save_memory(payload.text)
    return {"status": "ok"}


class MemoryRecentRequest(BaseModel):
    limit: int = 5


@router.post("/memory/recent")
def memory_recent(payload: MemoryRecentRequest):
    items = get_recent_memories(limit=payload.limit)
    return {"items": items}


