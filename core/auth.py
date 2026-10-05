from __future__ import annotations

import os
from typing import Optional

import requests
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from app.core.config import get_quota_config, get_security_config

_security = HTTPBearer()

ALGORITHM = "HS256"
SECRET_KEY = get_security_config()["jwt_secret"]


def create_token(subject: str, expires_delta: Optional[int] = None) -> str:
    payload = {"sub": subject}
    if expires_delta:
        import time

        payload["exp"] = int(time.time()) + int(expires_delta)
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return token


def verify_token(token: str) -> str:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        sub = payload.get("sub")
        if sub is None:
            raise JWTError("missing subject")
        return str(sub)
    except JWTError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))


def validate_billing_access(subject: str) -> None:
    """Optional billing gate for premium generation access. No-op unless configured."""
    billing_url = get_security_config()["billing_webhook_url"]
    if not billing_url:
        return
    try:
        resp = requests.post(billing_url, json={"subject": subject, "action": "generate"}, timeout=10)
        if resp.status_code == 402:
            raise HTTPException(status_code=status.HTTP_402_PAYMENT_REQUIRED, detail="Billing required")
        resp.raise_for_status()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"billing check failed: {exc}")


def require_auth(credentials: HTTPAuthorizationCredentials = Depends(_security)) -> str:
    scheme = credentials.scheme
    token = credentials.credentials
    if scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid auth scheme")
    subject = verify_token(token)
    check_user_quota(subject)
    validate_billing_access(subject)
    return subject


def get_user_tier(subject: str) -> str:
    """Resolve the user's current generation tier from an environment-driven policy."""
    tier = os.getenv("USER_TIER_OVERRIDE", "").strip()
    if tier:
        return tier.lower()
    if subject.endswith("@pro") or os.getenv("ENABLE_PRO_MODE", "false").strip().lower() in {"1", "true", "yes", "on"}:
        return "pro"
    return "standard"


def check_user_quota(subject: str) -> None:
    """Apply a simple quota policy for premium access without requiring a full billing system."""
    quota_cfg = get_quota_config()
    tier = get_user_tier(subject)
    quota_limit = quota_cfg["default_quota_per_minute"]
    if tier in {"pro", "premium", "enterprise"}:
        quota_limit = quota_cfg["pro_quota_per_minute"]
    if quota_cfg["billing_required"] and tier == "standard":
        raise HTTPException(status_code=status.HTTP_402_PAYMENT_REQUIRED, detail="Billing required for generation access")
    if os.getenv("SKIP_QUOTA_CHECK", "false").strip().lower() in {"1", "true", "yes", "on"}:
        return
    if quota_limit <= 0:
        return
    return
