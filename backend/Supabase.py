from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env", override=False)

try:
    from supabase import Client, create_client
except Exception:  # pragma: no cover - optional dependency during setup
    Client = Any  # type: ignore[misc,assignment]
    create_client = None  # type: ignore[assignment]


def get_supabase_config() -> Dict[str, str]:
    """Return environment-backed Supabase configuration."""
    return {
        "url": os.getenv("SUPABASE_URL", "").strip(),
        "anon_key": os.getenv("SUPABASE_ANON_KEY", "").strip(),
        "service_role_key": os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip(),
        "storage_bucket": os.getenv("SUPABASE_STORAGE_BUCKET", "normal-assets").strip(),
        "jwt_secret": os.getenv("SUPABASE_JWT_SECRET", "").strip(),
        "s3_endpoint": os.getenv("SUPABASE_S3_ENDPOINT", "").strip(),
        "generation_jobs_table": os.getenv("SUPABASE_GENERATION_JOBS", "generation_jobs").strip(),
        "generation_outputs_table": os.getenv("SUPABASE_GENERATION_OUTPUTS", "generation_outputs").strip(),
        "model_versions_table": os.getenv("SUPABASE_MODEL_VERSIONS", "model_versions").strip(),
        "training_runs_table": os.getenv("SUPABASE_TRAINING_RUNS", "training_runs").strip(),
        "safety_events_table": os.getenv("SUPABASE_SAFETY_EVENTS", "safety_events").strip(),
    }


def get_supabase_client() -> Optional[Client]:
    """Create a Supabase client when the required credentials are present."""
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY", "").strip()

    if not url or not key:
        return None
    if create_client is None:
        raise RuntimeError("python-supabase is not installed. Add 'supabase' to requirements.txt.")

    return create_client(url, key)


def write_record(table: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Insert a record into a configured Supabase table when the client is available."""
    client = get_supabase_client()
    if client is None:
        return None
    try:
        response = client.table(table).insert(payload).execute()
        data = getattr(response, "data", None)
        if data:
            return data[0]
        return None
    except Exception:
        return None


def log_analytics_event(event_name: str, **payload: Any) -> Optional[Dict[str, Any]]:
    """Persist a platform analytics event to Supabase using a standard table name."""
    row = {"event_name": event_name, "payload": payload}
    return write_record("analytics_events", row)


def log_feedback_event(event_name: str, **payload: Any) -> Optional[Dict[str, Any]]:
    """Persist a user feedback event for quality/monetization review."""
    row = {"event_name": event_name, "payload": payload}
    return write_record("feedback_events", row)


def log_model_event(event_name: str, **payload: Any) -> Optional[Dict[str, Any]]:
    """Persist a model lifecycle or rollout event."""
    row = {"event_name": event_name, "payload": payload}
    return write_record("model_events", row)


__all__ = [
    "get_supabase_config",
    "get_supabase_client",
    "write_record",
    "log_analytics_event",
    "log_feedback_event",
    "log_model_event",
]