"""Node orchestration for generation pipelines.

`NodeManager` composes an indexer, zero or more external connectors, and the
generator to produce outputs from raw user input. This is intentionally
framework-agnostic and focuses on a simple, testable pipeline that can be
expanded later with task graphs or node-based planners.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .indexer import SimpleIndexer
from .external import ExternalConnector
from .generator import generate_text


class NodeManager:
    def __init__(self, indexer: Optional[SimpleIndexer] = None, connectors: Optional[List[ExternalConnector]] = None):
        self.indexer = indexer or SimpleIndexer()
        self.connectors = connectors or []

    def run_pipeline(self, input_text: str, prompt_template: Optional[str] = None, top_k: int = 3) -> Dict:
        """Run a simple pipeline:

        1. Search the index for relevant documents
        2. Query external connectors for supporting data
        3. Build a composed prompt and generate output

        Returns a dict with `output`, `sources`, and `debug` information.
        """
        sources: List[Dict] = []

        # 1) index search
        try:
            hits = self.indexer.search(input_text, top_k=top_k)
        except Exception:
            hits = []

        for h in hits:
            sources.append({"type": "index", "doc": h})

        # 2) external connectors
        external_results: List[Dict] = []
        for c in self.connectors:
            try:
                res = c.fetch(params={"q": input_text})
            except Exception as e:
                res = {"error": str(e)}
            external_results.append({"connector": c.name, "result": res})

        # 3) compose prompt
        parts: List[str] = []
        if prompt_template:
            parts.append(prompt_template)
        if hits:
            parts.append("\n\nRelevant docs:\n" + "\n".join(f"- {d.get('id')}: {d.get('text')[:200]}" for d in hits))
        if external_results:
            parts.append("\n\nExternal results:\n" + "\n".join(f"- {r['connector']}: {str(r['result'])[:300]}" for r in external_results))

        parts.append(f"\n\nUser request: {input_text}\n\nResponse:")
        composed_prompt = "\n".join(parts)

        # generate
        try:
            output = generate_text(composed_prompt)
        except Exception as e:
            output = f"Generation failed: {e}"

        return {
            "output": output,
            "sources": sources,
            "external": external_results,
            "debug": {"prompt": composed_prompt[:2000]},
        }
