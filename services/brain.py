# AI logic engine

from __future__ import annotations


import os
from functools import lru_cache
from typing import List, Dict, Optional

from app.core.personas import PERSONAS
from app.core.prompts import build_persona_prompt
from app.services.memory import get_recent_memories
from app.generation.nodes import NodeManager
from app.generation.indexer import SimpleIndexer

import json
import requests

# Defer heavy imports; allow repo to run when `transformers`/`torch` are not installed.
_LOCAL_GEN_AVAILABLE = False
_PEFT_AVAILABLE = False
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _LOCAL_GEN_AVAILABLE = True
    try:
        from peft import PeftModel
        _PEFT_AVAILABLE = True
    except Exception:
        _PEFT_AVAILABLE = False
except Exception:
    _LOCAL_GEN_AVAILABLE = False


@lru_cache(maxsize=1)
def _get_local_generator():
    if not _LOCAL_GEN_AVAILABLE:
        raise RuntimeError("Local transformer backend not available")

    # Model tier mapping (no auto-download). Admin sets AI_MODEL_TIER or AI_MODEL_NAME.
    TIERS = {
        "EB": "distilgpt2",
        "B1": "gpt2",
        "F1": "facebook/opt-350m",
        "OFV": "facebook/opt-1.3b",
        "GEM": "facebook/opt-6.7b",
        "GEMSTONE": "facebook/opt-13b",
        "OMNI": os.getenv("AI_MODEL_NAME", "facebook/opt-6.7b"),
    }

    tier = os.getenv("AI_MODEL_TIER", "EB").upper()
    model_name = os.getenv("AI_MODEL_NAME") or TIERS.get(tier, "distilgpt2")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Load base model (CPU-friendly defaults)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # If there's a LoRA adapter specified and PEFT is installed, load it.
    lora_path = os.getenv("AI_LORA_PATH")
    if lora_path and _PEFT_AVAILABLE:
        try:
            model = PeftModel.from_pretrained(model, lora_path)
        except Exception:
            # ignore and continue with base model
            pass

    # Ensure tokenizer has a pad token
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Inference config
    max_new_tokens = int(os.getenv("AI_MAX_NEW_TOKENS", "120"))
    temperature = float(os.getenv("AI_TEMPERATURE", "0.9"))
    top_p = float(os.getenv("AI_TOP_P", "0.95"))

    return {
        "tokenizer": tokenizer,
        "model": model,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }


def _remote_generate(prompt: str, model_name: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    """Use Hugging Face Inference API as a fallback when local transformers are unavailable.

    Requires `HF_API_TOKEN` environment variable.
    """
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

    # HF may return list or dict depending on model
    if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
        return data[0]["generated_text"]
    if isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"]

    # Fallback: try raw text
    return str(data)


def generate_response(message: str, persona: Dict, persona_key: Optional[str] = None) -> str:
    message = (message or "").strip()
    if not message:
        return "I need a message to respond."

    # Determine tier: persona override (Telvin -> OMNI), then env var, then default EB
    env_tier = os.getenv("AI_MODEL_TIER")
    if persona_key and str(persona_key).lower() == "telvin":
        tier = "OMNI"
    else:
        tier = env_tier or "EB"

    system_prompt = build_persona_prompt(persona, tier=tier)

    memories: List[str] = get_recent_memories(limit=int(os.getenv("AI_MEMORY_LIMIT", "5")))
    memory_block = ""
    if memories:
        memory_block = "\n\nRelevant memories:\n" + "\n".join(f"- {m}" for m in memories)

    prompt = f"{system_prompt}{memory_block}\n\nUser: {message}\n{persona['name']}:"

    model_name = os.getenv("AI_MODEL_NAME")
    max_new_tokens = int(os.getenv("AI_MAX_NEW_TOKENS", "120"))
    temperature = float(os.getenv("AI_TEMPERATURE", "0.9"))
    top_p = float(os.getenv("AI_TOP_P", "0.95"))

    # Optionally run a retrieval-augmented generation pipeline when enabled
    use_rag = str(os.getenv("AI_USE_RAG", "false")).lower() in ("1", "true", "yes")
    if use_rag:
        try:
            index_name = os.getenv("AI_INDEX_NAME", "default")
            indexer = SimpleIndexer(name=index_name)
            node_manager = NodeManager(indexer=indexer, connectors=[])
            nm_result = node_manager.run_pipeline(message, prompt_template=system_prompt)
            text = nm_result.get("output") or None
        except Exception:
            text = None

    # Try local generation first (preferred for no-API deployments) if RAG not used or failed
    if not text and _LOCAL_GEN_AVAILABLE:
        try:
            gen = _get_local_generator()
            tokenizer = gen["tokenizer"]
            model = gen["model"]
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            # Force CPU for environments without GPU
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
        except Exception:
            # fall through to remote option
            text = None
    elif not text:
        text = None

    # If local failed or not available, try remote HF Inference API
    if not text:
        try:
            text = _remote_generate(prompt, model_name, max_new_tokens, temperature, top_p)
        except Exception as e:
            return (
                "Model generation failed. Ensure you have either:\n"
                "- installed `transformers` and a compatible `torch` wheel for your Python version, or\n"
                "- set the `HF_API_TOKEN` environment variable to use Hugging Face Inference API.\n"
                f"Details: {e}"
            )

    # Heuristic: return only the tail after the last persona marker
    marker = f"{persona['name']}:"
    if marker in text:
        return text.split(marker)[-1].strip()

    # Fallback
    return text.strip()

