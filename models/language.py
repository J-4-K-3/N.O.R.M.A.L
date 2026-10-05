"""Language utilities: summarization, entity extraction, transformations."""

from __future__ import annotations

from typing import Dict, List, Optional
from models.model_manager import get_manager
from models.lang_tokenizer import detect_language, get_language_context


def summarize(text: str, max_length: int = 200) -> str:
    mgr = get_manager()
    ctx = get_language_context(text)
    lang = ctx.get("language", "en")
    # ask the model to summarize and preserve language
    prompt = f"Summarize in {max_length} chars. Preserve language {lang}. Text: {text}"
    return mgr.generate_text(prompt)


def extract_entities(text: str) -> List[Dict[str, str]]:
    mgr = get_manager()
    ctx = get_language_context(text)
    prompt = f"Extract named entities from the following text (language={ctx.get('language')}): {text}"
    resp = mgr.generate_text(prompt)
    # naive parse placeholder — returns raw model output as one entry
    return [{"raw": resp}]


def paraphrase(text: str, style: Optional[str] = None) -> str:
    mgr = get_manager()
    ctx = get_language_context(text)
    style_note = f" in style: {style}" if style else ""
    prompt = f"Paraphrase the following (language={ctx.get('language')}){style_note}: {text}"
    return mgr.generate_text(prompt)
