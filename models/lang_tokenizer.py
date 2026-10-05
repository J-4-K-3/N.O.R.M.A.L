"""Lightweight language detector and tokenizer utilities.

Provides `detect_language`, `tokenize`, and `get_language_context` helpers.
Designed to be fast and dependency-light; heavy NLP tokenizers can be
plugged in later per-language if needed.
"""

from __future__ import annotations

import re
from typing import Dict, List

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
except Exception:
    detect = None

# basic CJK range test
_CJK_RE = re.compile(r"[\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF]")


def detect_language(text: str) -> str:
    """Return a 2-letter language code or 'und' if unknown."""
    if not text or not text.strip():
        return "und"
    if detect:
        try:
            return detect(text)
        except Exception:
            pass
    # fallback heuristic: CJK detection
    if _CJK_RE.search(text):
        return "zh"
    return "en"


def _cjk_tokenize(text: str) -> List[str]:
    # simple char-level tokens for CJK languages
    return [c for c in text if not c.isspace()]


def _simple_tokenize(text: str) -> List[str]:
    # split on word boundaries; keep alphanumerics
    tokens = re.findall(r"\b\w+\b", text, flags=re.UNICODE)
    return tokens


def tokenize(text: str) -> List[str]:
    """Return a list of tokens suitable for quick language-aware processing."""
    if not text:
        return []
    if _CJK_RE.search(text):
        return _cjk_tokenize(text)
    return _simple_tokenize(text)


def get_language_context(text: str, max_tokens: int = 50) -> Dict:
    """Return a small context dict: detected language, token count, sample tokens."""
    lang = detect_language(text)
    toks = tokenize(text)
    return {"language": lang, "token_count": len(toks), "sample_tokens": toks[:max_tokens]}
