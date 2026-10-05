"""Simple generation utilities.

This module provides a lightweight `generate_text` function that prefers a local
transformers backend if available (via `app.services.brain`) and falls back to
Hugging Face Inference API when configured via `HF_API_TOKEN`.

The implementation mirrors the engine's existing approach but exposes a
clean, testable interface for downstream orchestration.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import requests

try:
    from app.services import brain as _brain
except Exception:
    _brain = None


def _remote_generate(prompt: str, model_name: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    hf_token = os.getenv("HF_API_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_API_TOKEN not set for remote generation")

    url = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "return_full_text": True,
        },
    }

    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
        return data[0]["generated_text"]
    if isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"]

    return str(data)


def generate_text(
    prompt: str,
    model_name: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    prefer_local: bool = True,
) -> str:
    """Generate text from a prompt.

    Behavior:
    - If a local transformers backend is available via `app.services.brain` and
      `prefer_local` is True, use it.
    - Otherwise, use Hugging Face Inference API if `HF_API_TOKEN` is present.

    This function intentionally mirrors the config-driven defaults used by
    the rest of the engine.
    """
    model_name = model_name or os.getenv("AI_MODEL_NAME", "distilgpt2")
    max_new_tokens = int(max_new_tokens or os.getenv("AI_MAX_NEW_TOKENS", "120"))
    temperature = float(temperature or os.getenv("AI_TEMPERATURE", "0.9"))
    top_p = float(top_p or os.getenv("AI_TOP_P", "0.95"))

    # Try local generation
    if prefer_local and _brain is not None and getattr(_brain, "_LOCAL_GEN_AVAILABLE", False):
        try:
            gen = _brain._get_local_generator()
            tokenizer = gen["tokenizer"]
            model = gen["model"]
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            device = "cpu"
            try:
                input_ids = input_ids.to(device)
                model.to(device)
            except Exception:
                pass

            output_ids = model.generate(
                input_ids,
                max_new_tokens=gen["max_new_tokens"],
                do_sample=True,
                temperature=gen["temperature"],
                top_p=gen["top_p"],
                pad_token_id=tokenizer.pad_token_id,
            )

            text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return text.strip()
        except Exception:
            # fall through to remote
            pass

    # Remote generation fallback
    return _remote_generate(prompt, model_name, max_new_tokens, temperature, top_p)
