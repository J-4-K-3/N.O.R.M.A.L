from __future__ import annotations

from typing import Dict
from .celery_app import celery_app

from app.generation.indexer import SimpleIndexer
from app.generation.nodes import NodeManager


@celery_app.task(bind=True)
def generate_job(self, input_text: str, index_name: str = "default", user: str | None = None) -> Dict:
    indexer = SimpleIndexer(name=index_name)
    node_manager = NodeManager(indexer=indexer, connectors=[])
    result = node_manager.run_pipeline(input_text)
    payload = {"output": result.get("output"), "debug": result.get("debug", {})}
    if user is not None:
        payload["user"] = user
    return payload
