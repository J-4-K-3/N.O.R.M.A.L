from __future__ import annotations

import os
from celery import Celery

broker = os.getenv("CELERY_BROKER_URL", os.getenv("REDIS_URL", "redis://localhost:6379/0"))
backend = os.getenv("CELERY_RESULT_BACKEND", broker)

celery_app = Celery("normal_engine", broker=broker, backend=backend)

celery_app.conf.task_routes = {
    "app.tasks.generation_tasks.*": {"queue": "generation"}
}
