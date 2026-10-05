"""A simple placeholder indexer.

This `SimpleIndexer` is intentionally lightweight: it stores documents as JSON
files under a directory and implements a naive substring-based search. It's a
scaffold for integrating a real embedding-based vector index later.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional


class SimpleIndexer:
    def __init__(self, name: str = "default", index_dir: Optional[str] = None):
        self.name = name
        self.index_dir = index_dir or os.path.join("data", "indices", name)
        os.makedirs(self.index_dir, exist_ok=True)
        self._meta_path = os.path.join(self.index_dir, "index.json")
        if not os.path.exists(self._meta_path):
            with open(self._meta_path, "w", encoding="utf-8") as fh:
                json.dump({"documents": []}, fh)

    def index_documents(self, documents: List[Dict[str, str]]) -> None:
        """Index documents.

        documents: list of {"id": str, "text": str, "meta": {...}}
        """
        data = self._read_meta()
        data_docs = data.get("documents", [])
        # append new docs (no dedupe in this simple implementation)
        data_docs.extend(documents)
        data["documents"] = data_docs
        self._write_meta(data)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, str]]:
        """Naive substring search that returns the best-scoring documents."""
        data = self._read_meta()
        docs = data.get("documents", [])
        results = []
        q = (query or "").lower()
        for d in docs:
            text = (d.get("text") or "").lower()
            score = 0
            if not q:
                score = 0
            else:
                score = text.count(q)
                # small boost for prefix match
                if text.startswith(q):
                    score += 1

            if score > 0:
                results.append({"doc": d, "score": score})

        results.sort(key=lambda x: x["score"], reverse=True)
        return [r["doc"] for r in results[:top_k]]

    def _read_meta(self) -> Dict:
        with open(self._meta_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _write_meta(self, data: Dict) -> None:
        with open(self._meta_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
