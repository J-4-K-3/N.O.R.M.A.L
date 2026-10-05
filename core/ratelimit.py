from __future__ import annotations

import os
import time
from typing import Dict

try:
    import redis
except Exception:
    redis = None

from fastapi import HTTPException, status

from app.core.config import get_quota_config

# Redis is the distributed enforcement layer; in-memory is a dev/single-process fallback.
_IN_MEMORY: Dict[str, Dict[str, int]] = {}
RATE_LIMIT = int(os.getenv("API_RATE_LIMIT", "60"))


def _now_minute() -> int:
    return int(time.time() // 60)


def _limit_for_key(key: str) -> int:
    return max(1, int(os.getenv(f"API_RATE_LIMIT_{key.upper()}", RATE_LIMIT)))


def _user_quota_limit(key: str) -> int:
    quota_cfg = get_quota_config()
    tier = os.getenv(f"USER_TIER_{key.upper()}", "standard").lower()
    if tier in {"pro", "premium", "enterprise"}:
        return quota_cfg["pro_quota_per_minute"]
    return quota_cfg["default_quota_per_minute"]


def check_rate_limit(key: str) -> None:
    """Apply a per-user / per-key rate limit for the current minute."""
    limit = _limit_for_key(key)
    if os.getenv("SKIP_QUOTA_CHECK", "false").strip().lower() in {"1", "true", "yes", "on"}:
        return
    limit = min(limit, _user_quota_limit(key))

    if redis is not None and os.getenv("REDIS_URL"):
        try:
            client = redis.from_url(os.getenv("REDIS_URL"))
            field = str(_now_minute())
            cur = client.hincrby(f"ratelimit:{key}", field, 1)
            client.expire(f"ratelimit:{key}", 3600)
            if cur > limit:
                raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="rate limit exceeded")
            return
        except Exception as exc:
            if hasattr(redis, "exceptions") and isinstance(exc, redis.exceptions.RedisError):
                pass
            else:
                raise

    minute = _now_minute()
    entry = _IN_MEMORY.get(key)
    if not entry or entry.get("minute") != minute:
        _IN_MEMORY[key] = {"minute": minute, "count": 1}
        return
    entry["count"] += 1
    if entry["count"] > limit:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="rate limit exceeded")
