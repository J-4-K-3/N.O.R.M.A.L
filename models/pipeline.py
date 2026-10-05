"""Orchestration pipeline for INPUT -> ENGINE -> THINKING -> REASONING -> RESPONSE

This module wires together `ModelManager`, `reasoning`, `vision_api`, and
`language` utilities into a single Pipeline class that can be extended with
tooling and branching logic.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from models.model_manager import get_manager
from models import reasoning
from models import vision_api
from models import language
from models.embeddings import get_embedding_service
import os


class Pipeline:
    def __init__(self):
        self.manager = get_manager()
        self.vision = vision_api.get_vision_api()

    def run(self, input_text: str, image_bytes: Optional[bytes] = None) -> Dict[str, Any]:
        # ENGINE: normalize input
        engine_ctx = {"input": input_text}
        if image_bytes:
            vis = self.vision.describe_image(image_bytes)
            engine_ctx["vision"] = vis

        # THINKING: generate hypotheses or chains of thought
        thinking = reasoning.chain_of_thought(input_text, steps=2)
        engine_ctx["thinking"] = thinking

        # Branching: retrieval-augmented generation (if enabled)
        use_rag = str(os.getenv("AI_USE_RAG", "false")).lower() in ("1", "true", "yes")
        retrieved: Optional[list] = None
        if use_rag:
            emb = get_embedding_service()
            hits = emb.search(input_text, top_k=5)
            retrieved = hits
            engine_ctx["retrieved"] = hits

        # REASONING: decompose and solve
        # Optionally include retrieved docs in the reasoning context
        reasoning_input = input_text
        if retrieved:
            # include ids/scores in the prompt (full docs can be pulled from an index)
            retrieved_block = "\n\nRetrieved:\n" + "\n".join(f"- {r[0]} (score={r[1]:.3f})" for r in retrieved)
            reasoning_input = f"{input_text}{retrieved_block}"

        reasoning_out = reasoning.decompose_and_solve(reasoning_input)
        engine_ctx["reasoning"] = reasoning_out

        # RESPONSE: produce final natural-language answer
        final = self.manager.generate_text(f"Using reasoning:\n{reasoning_out}\nProduce a concise response to: {input_text}")

        return {"final": final, "ctx": engine_ctx}


def run_pipeline(input_text: str, image_bytes: Optional[bytes] = None) -> Dict[str, Any]:
    p = Pipeline()
    return p.run(input_text, image_bytes=image_bytes)
