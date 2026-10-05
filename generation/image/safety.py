from __future__ import annotations

import base64
import os
from typing import Any, Dict, List, Optional

import requests


_POLICY_GROUPS: Dict[str, List[str]] = {
    "sexual_content": [
        "nude",
        "nudity",
        "sex",
        "sexual",
        "explicit",
        "porn",
        "pornographic",
    ],
    "graphic_violence": [
        "graphic",
        "gore",
        "violence",
        "blood",
        "murder",
        "kill",
        "beheading",
    ],
    "self_harm": [
        "suicide",
        "self-harm",
        "self harm",
        "cutting",
        "bleeding out",
    ],
    "hate_or_abuse": [
        "slur",
        "hate",
        "dehumanize",
        "abuse",
        "attack someone",
    ],
}


_COPYRIGHT_PATTERNS: List[str] = [
    "in the style of",
    "inspired by",
    "remix of",
    "cover of",
    "song by",
    "album by",
    "as performed by",
    "imitating",
    "copy of",
    "exactly like",
]


def _normalize_prompt(prompt: str) -> str:
    return (prompt or "").strip().lower()


def _group_hits(normalized_prompt: str, groups: Dict[str, List[str]]) -> Dict[str, List[str]]:
    hits: Dict[str, List[str]] = {}
    for category, keywords in groups.items():
        matches = [keyword for keyword in keywords if keyword in normalized_prompt]
        if matches:
            hits[category] = matches
    return hits


def _risk_score_from_group_hits(group_hits: Dict[str, List[str]]) -> float:
    if not group_hits:
        return 0.0
    severity_weight = {
        "sexual_content": 0.6,
        "graphic_violence": 0.8,
        "self_harm": 0.9,
        "hate_or_abuse": 0.85,
    }
    score = 0.0
    for category, matches in group_hits.items():
        category_score = severity_weight.get(category, 0.5) * min(1.0, len(matches) / 3.0)
        score = max(score, category_score)
    return min(1.0, score)


def _severity_from_score(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


def check_prompt_safety(prompt: str) -> Dict[str, Any]:
    """Run a lightweight local policy gate for obvious unsafe prompt text."""
    normalized = _normalize_prompt(prompt)
    group_hits = _group_hits(normalized, _POLICY_GROUPS)
    if not group_hits:
        return {
            "safe": True,
            "reason": "policy_pass",
            "hits": [],
            "categories": {},
            "score": 0.0,
            "severity": "low",
            "classifier": "keyword_policy_gate",
        }

    flattened_hits = [hit for values in group_hits.values() for hit in values]
    score = _risk_score_from_group_hits(group_hits)
    return {
        "safe": False,
        "reason": "unsafe_prompt_keywords",
        "hits": flattened_hits,
        "categories": group_hits,
        "score": round(score, 3),
        "severity": _severity_from_score(score),
        "classifier": "keyword_policy_gate",
    }


def check_image_safety_bytes(data: bytes, prompt: Optional[str] = None) -> Dict[str, Any]:
    """Run a safety check on image bytes.

    This implementation uses a lightweight prompt policy gate first and then a remote safety API when configured.
    If the remote safety service fails, the result fails closed instead of permitting generation.
    """
    prompt_result = check_prompt_safety(prompt or "")
    if not prompt_result["safe"]:
        return {
            "safe": False,
            "score": float(prompt_result["score"]),
            "reason": prompt_result["reason"],
            "hits": prompt_result["hits"],
            "categories": prompt_result["categories"],
            "severity": prompt_result["severity"],
            "classifier": prompt_result["classifier"],
        }

    api = os.getenv("SAFETY_API_URL")
    if api:
        payload = {"image": base64.b64encode(data).decode("utf-8")}
        try:
            resp = requests.post(api, json=payload, timeout=15)
            resp.raise_for_status()
            result = resp.json()
            if "safe" not in result:
                result["safe"] = True
            result.setdefault("classifier", "external_safety_api")
            return result
        except Exception as exc:
            return {
                "safe": False,
                "score": 0.0,
                "local_fallback": True,
                "reason": "safety_service_unavailable",
                "error": str(exc),
                "requires_review": True,
                "classifier": "external_safety_api",
            }

    return {
        "safe": True,
        "score": 0.0,
        "local_fallback": True,
        "reason": "local_policy_pass",
        "classifier": "keyword_policy_gate",
    }


def check_copyright_risk(prompt: str, allow_copyrighted: bool = False) -> Dict[str, Any]:
    """Detect prompt patterns commonly associated with requests to imitate or reproduce copyrighted works."""
    if allow_copyrighted:
        return {
            "safe": True,
            "reason": "copyright_override_allowed",
            "matches": [],
            "score": 0.0,
            "classifier": "copyright_pattern_gate",
        }

    normalized = _normalize_prompt(prompt)
    matches = [pattern for pattern in _COPYRIGHT_PATTERNS if pattern in normalized]
    if matches:
        return {
            "safe": False,
            "reason": "copyright_risk",
            "matches": matches,
            "score": 1.0,
            "severity": "high",
            "classifier": "copyright_pattern_gate",
        }

    return {
        "safe": True,
        "reason": "copyright_policy_pass",
        "matches": [],
        "score": 0.0,
        "severity": "low",
        "classifier": "copyright_pattern_gate",
    }


def check_prompt_policy(prompt: str, allow_copyrighted: bool = False) -> Dict[str, Any]:
    """Combine prompt-risk and copyright heuristics into a single enterprise-friendly policy result."""
    prompt_guard = check_prompt_safety(prompt)
    copyright_guard = check_copyright_risk(prompt, allow_copyrighted=allow_copyrighted)

    combined = {
        "safe": prompt_guard["safe"] and copyright_guard["safe"],
        "score": max(float(prompt_guard.get("score", 0.0)), float(copyright_guard.get("score", 0.0))),
        "severity": "high" if max(float(prompt_guard.get("score", 0.0)), float(copyright_guard.get("score", 0.0))) >= 0.8 else "medium" if max(float(prompt_guard.get("score", 0.0)), float(copyright_guard.get("score", 0.0))) >= 0.4 else "low",
        "prompt": prompt_guard,
        "copyright": copyright_guard,
        "requires_review": not (prompt_guard["safe"] and copyright_guard["safe"]),
    }

    if not combined["safe"]:
        combined["reason"] = "policy_block"
    else:
        combined["reason"] = "policy_pass"

    return combined
