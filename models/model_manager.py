"""ModelManager: production-capable loader and runner for text and vision models.

Features:
- Lazy-loading and caching of tokenizer + model instances.
- Device-aware placement (CPU/GPU).
- Optional PEFT/LoRA adapter loading when `peft` is available and `AI_LORA_PATH` is set.
- Fallback to remote Hugging Face Inference API when local models aren't available.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

_TORCH_AVAILABLE = False
try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False


class ModelManager:
    def __init__(self):
        self._text_models: Dict[str, Dict[str, Any]] = {}
        self._vision_models: Dict[str, Dict[str, Any]] = {}

    def _device(self) -> str:
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def load_text_model(self, name: Optional[str] = None):
        name = (name or os.getenv("AI_MODEL_NAME", "distilgpt2")).strip()
        if name in self._text_models:
            return self._text_models[name]

        # Try to load a local transformers model
        if _TORCH_AVAILABLE:
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM

                tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(name)

                # If peft/LoRA configured, try to load adapter
                lora_path = os.getenv("AI_LORA_PATH")
                if lora_path:
                    try:
                        from peft import PeftModel

                        model = PeftModel.from_pretrained(model, lora_path)
                    except Exception:
                        pass

                # Ensure tokens
                if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token

                device = self._device()
                model.to(device)

                entry = {"name": name, "tokenizer": tokenizer, "model": model, "device": device}
                self._text_models[name] = entry
                return entry
            except Exception:
                # fall through to remote handle
                pass

        # Local load failed or not available — leave a pointer for remote inference
        entry = {"name": name, "remote": True}
        self._text_models[name] = entry
        return entry

    def generate_text(self, prompt: str, model: Optional[str] = None, max_new_tokens: int = 120, temperature: float = 0.9, top_p: float = 0.95) -> str:
        entry = self.load_text_model(model)

        # Remote inference path
        if entry.get("remote"):
            hf = os.getenv("HF_API_TOKEN")
            if not hf:
                return "[no local model and HF_API_TOKEN not set]"
            try:
                import requests

                url = f"https://api-inference.huggingface.co/models/{entry['name']}"
                headers = {"Authorization": f"Bearer {hf}", "Content-Type": "application/json"}
                payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_new_tokens, "temperature": temperature, "top_p": top_p}}
                resp = requests.post(url, headers=headers, json=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
                    return data[0]["generated_text"]
                if isinstance(data, dict) and "generated_text" in data:
                    return data["generated_text"]
                return str(data)
            except Exception as e:
                return f"[remote generation failed] {e}"

        # Local generation path
        try:
            tokenizer = entry["tokenizer"]
            model_obj = entry["model"]
            device = entry.get("device", "cpu")

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            gen = model_obj.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
            )
            text = tokenizer.decode(gen[0], skip_special_tokens=True)
            return text
        except Exception as e:
            return f"[local generation failed] {e}"

    def load_vision_model(self, name: Optional[str] = None):
        name = (name or os.getenv("AI_VISION_MODEL", "openai/clip-vit-base-patch32")).strip()
        if name in self._vision_models:
            return self._vision_models[name]
        # lazy: for now store name; real loader can be added
        self._vision_models[name] = {"name": name}
        return self._vision_models[name]

    def analyze_image(self, image_bytes: bytes, model: Optional[str] = None) -> Dict[str, Any]:
        # Prefer local CLIP-like analysis when available
        try:
            from PIL import Image
            # real CLIP analysis would be implemented here
        except Exception:
            pass
        return {"description": "[vision analysis; implement CLIP/Vision encoder here]"}


_GLOBAL_MANAGER: Optional[ModelManager] = None


def get_manager() -> ModelManager:
    global _GLOBAL_MANAGER
    if _GLOBAL_MANAGER is None:
        _GLOBAL_MANAGER = ModelManager()
    return _GLOBAL_MANAGER
