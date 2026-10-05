"""EmbeddingService: computes embeddings and provides a FAISS vector index.

Uses `sentence-transformers` for embeddings and `faiss` for vector search.
Indices are stored under `data/indices/<name>/` with `meta.json` and `index.faiss`.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import faiss
except Exception:
    faiss = None

import numpy as np


class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", index_name: str = "default"):
        self.model_name = model_name
        self.index_name = index_name
        self.index_dir = os.path.join("data", "indices", index_name)
        os.makedirs(self.index_dir, exist_ok=True)
        self._meta_path = os.path.join(self.index_dir, "meta.json")
        self._index_path = os.path.join(self.index_dir, "index.faiss")

        self._model = None
        self._index = None
        self._ids: List[str] = []
        self._dim = None

        self._load()

    def _load(self):
        # load meta
        if os.path.exists(self._meta_path):
            with open(self._meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
                self._ids = meta.get("ids", [])
                self._dim = meta.get("dim")

        # init model
        if SentenceTransformer is not None:
            try:
                self._model = SentenceTransformer(self.model_name)
                self._dim = self._dim or self._model.get_sentence_embedding_dimension()
            except Exception:
                self._model = None

        # init or load FAISS index
        if faiss is not None and self._dim:
            if os.path.exists(self._index_path):
                try:
                    self._index = faiss.read_index(self._index_path)
                except Exception:
                    self._index = faiss.IndexFlatIP(self._dim)
            else:
                self._index = faiss.IndexFlatIP(self._dim)

    def _save_meta(self):
        with open(self._meta_path, "w", encoding="utf-8") as fh:
            json.dump({"ids": self._ids, "dim": self._dim}, fh)

    def index_documents(self, documents: List[Dict[str, str]]) -> int:
        """Index list of documents: each doc = {"id": str, "text": str, "meta": {...}}"""
        texts = [d.get("text", "") for d in documents]
        ids = [d.get("id") for d in documents]
        if not self._model:
            raise RuntimeError("Embedding model not available; install sentence-transformers")
        embs = self._model.encode(texts, convert_to_numpy=True)
        # normalize for cosine similarity with inner product
        embs = embs.astype("float32")
        faiss.normalize_L2(embs)

        if self._index is None:
            self._dim = embs.shape[1]
            if faiss is None:
                raise RuntimeError("faiss not available")
            self._index = faiss.IndexFlatIP(self._dim)

        self._index.add(embs)
        self._ids.extend(ids)
        self._save_meta()
        # save index
        try:
            faiss.write_index(self._index, self._index_path)
        except Exception:
            pass
        return len(ids)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if not self._model or self._index is None:
            return []
        q_emb = self._model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_emb)
        D, I = self._index.search(q_emb, top_k)
        results: List[Tuple[str, float]] = []
        for idx, score in zip(I[0], D[0]):
            if idx < 0 or idx >= len(self._ids):
                continue
            results.append((self._ids[idx], float(score)))
        return results

    def get_meta_for_id(self, id_value: str) -> Optional[Dict]:
        # naive lookup: load index meta mapping (not storing full docs here)
        # For full metadata, store a separate meta store or use SimpleIndexer
        return None


_GLOBAL_EMB: Optional[EmbeddingService] = None


def get_embedding_service(name: str = "default") -> EmbeddingService:
    global _GLOBAL_EMB
    if _GLOBAL_EMB is None:
        _GLOBAL_EMB = EmbeddingService(index_name=name)
    return _GLOBAL_EMB
