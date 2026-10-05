from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import get_experiment_config


def _get_log_path(name: str = "app") -> Path:
    cfg = get_experiment_config()
    log_dir = Path(cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{name}.log.jsonl"


def log_event(event_type: str, message: str, **context: Any) -> Dict[str, Any]:
    """Append a structured JSON log entry for operational tracing."""
    entry: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "message": message,
    }
    entry.update(context)
    log_path = _get_log_path(context.pop("log_name", "app"))
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def log_generation_event(task_id: Optional[str], status: str, **context: Any) -> Dict[str, Any]:
    payload = {"task_id": task_id, "status": status}
    payload.update(context)
    return log_event("generation", f"generation task {status}", **payload)


def log_training_event(run_name: str, status: str, **context: Any) -> Dict[str, Any]:
    payload = {"run_name": run_name, "status": status}
    payload.update(context)
    return log_event("training", f"training run {status}", **payload)


def log_safety_event(event_type: str, status: str, **context: Any) -> Dict[str, Any]:
    payload = {"status": status, "event_type": event_type}
    payload.update(context)
    return log_event("safety", f"safety {status}", **payload)


def read_recent_events(limit: int = 25, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """Read recent structured events from the JSONL log files and return them as dicts."""
    log_dir = Path(get_experiment_config()["log_dir"])
    if not log_dir.exists():
        return []

    events: List[Dict[str, Any]] = []
    for log_file in sorted(log_dir.glob("*.log.jsonl")):
        with log_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event_type and item.get("event_type") != event_type:
                    continue
                events.append(item)
    events.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return events[:limit]


def summarize_events(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize a list of log events into counts by event type and a recent activity tail."""
    counts: Dict[str, int] = {}
    for event in events:
        name = event.get("event_type", "unknown")
        counts[name] = counts.get(name, 0) + 1
    return {
        "total": len(events),
        "counts_by_event_type": counts,
        "recent_events": events[:10],
    }


def log_analytics_event(event_name: str, **context: Any) -> Dict[str, Any]:
    """Record a platform analytics event and return the created log object."""
    payload = {"event_name": event_name}
    payload.update(context)
    try:
        from backend.Supabase import log_analytics_event as supabase_log

        supabase_log(event_name, **context)
    except Exception:
        pass
    return log_event("analytics", f"analytics event: {event_name}", **payload)


def log_feedback_event(event_name: str, **context: Any) -> Dict[str, Any]:
    """Record a user feedback event for experimentation and quality review."""
    payload = {"event_name": event_name}
    payload.update(context)
    try:
        from backend.Supabase import log_feedback_event as supabase_log

        supabase_log(event_name, **context)
    except Exception:
        pass
    return log_event("feedback", f"feedback event: {event_name}", **payload)
