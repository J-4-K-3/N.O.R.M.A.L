# settings, env, constants
from __future__ import annotations

from typing import Dict, Any
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env", override=False)


def get_supabase_config() -> Dict[str, str]:
    """Return Supabase runtime settings used by the generation backend."""
    return {
        "url": os.getenv("SUPABASE_URL", "").strip(),
        "anon_key": os.getenv("SUPABASE_ANON_KEY", "").strip(),
        "service_role_key": os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip(),
        "storage_bucket": os.getenv("SUPABASE_STORAGE_BUCKET", "normal-assets").strip(),
        "jwt_secret": os.getenv("SUPABASE_JWT_SECRET", "").strip(),
    }


ENGINE_IDENTITY = {
    "name": "N.O.R.M.A.L",
    "full_name": "Neural Optimized Reasoning Machine for Advanced Learning",
    "organization": "Innoxation Tech Inc.",
    "ceo": "Jacob",
    "version": "1.0.0",
    "runtime": "local-transformers",
    "brand": "Innoxation AI Core",
}

MODEL_TIERS: Dict[str, Dict[str, Any]] = {
    "EB": {
        "label": "EB",
        "name": "Beta Edition",
        "description": "lightweight conversational model",
        "context_window": 2048,
        "max_new_tokens": 120,
        "temperature": 0.8,
        "top_p": 0.9,
        "system_behavior": "concise, friendly, reliable",
        "model": os.getenv("AI_MODEL_NAME", "distilgpt2"),
        "tools": ["chat"],
        "capabilities": ["chat", "basic_reasoning"],
    },
    "B1": {
        "label": "B1",
        "name": "First Beta",
        "description": "general assistant tier",
        "context_window": 4096,
        "max_new_tokens": 180,
        "temperature": 0.85,
        "top_p": 0.93,
        "system_behavior": "helpful, structured, conversational",
        "model": os.getenv("AI_MODEL_NAME_B1", os.getenv("AI_MODEL_NAME", "gpt2")),
        "tools": ["chat", "memory"],
        "capabilities": ["chat", "memory", "basic_reasoning"],
    },
    "F1": {
        "label": "F1",
        "name": "First Flash",
        "description": "faster reasoning and code assistance",
        "context_window": 8192,
        "max_new_tokens": 220,
        "temperature": 0.8,
        "top_p": 0.94,
        "system_behavior": "smart, concise, product-minded",
        "model": os.getenv("AI_MODEL_NAME_F1", os.getenv("AI_MODEL_NAME", "facebook/opt-350m")),
        "tools": ["chat", "memory", "code_generation"],
        "capabilities": ["chat", "memory", "code_generation", "tool_use"],
    },
    "OFV": {
        "label": "OFV",
        "name": "Official Version",
        "description": "production-grade reasoning and project support",
        "context_window": 16384,
        "max_new_tokens": 300,
        "temperature": 0.75,
        "top_p": 0.95,
        "system_behavior": "professional, exact, well-structured",
        "model": os.getenv("AI_MODEL_NAME_OFV", os.getenv("AI_MODEL_NAME", "facebook/opt-1.3b")),
        "tools": ["chat", "memory", "code_generation", "project_planning"],
        "capabilities": ["chat", "memory", "code_generation", "planning", "tool_use"],
    },
    "GEM": {
        "label": "GEM",
        "name": "Gem",
        "description": "high-capacity public reasoning model",
        "context_window": 32768,
        "max_new_tokens": 400,
        "temperature": 0.7,
        "top_p": 0.96,
        "system_behavior": "analytical, strategic, business-aware",
        "model": os.getenv("AI_MODEL_NAME_GEM", os.getenv("AI_MODEL_NAME", "facebook/opt-6.7b")),
        "tools": ["chat", "memory", "code_generation", "project_planning", "analysis"],
        "capabilities": ["chat", "memory", "code_generation", "analysis", "planning", "tool_use"],
    },
    "GEMSTONE": {
        "label": "GEMSTONE",
        "name": "Gemstone",
        "description": "advanced public intelligence with stronger reasoning",
        "context_window": 32768,
        "max_new_tokens": 500,
        "temperature": 0.65,
        "top_p": 0.97,
        "system_behavior": "deep reasoning, precise output, advanced synthesis",
        "model": os.getenv("AI_MODEL_NAME_GEMSTONE", os.getenv("AI_MODEL_NAME", "facebook/opt-13b")),
        "tools": ["chat", "memory", "code_generation", "project_planning", "analysis", "multi_step_reasoning"],
        "capabilities": ["chat", "memory", "code_generation", "analysis", "planning", "multi_step_reasoning", "tool_use"],
    },
    "OMNI": {
        "label": "OMNI",
        "name": "Omni",
        "description": "highest internal model for Innoxation operations",
        "context_window": 65536,
        "max_new_tokens": 700,
        "temperature": 0.6,
        "top_p": 0.98,
        "system_behavior": "elite reasoning, executive-level planning, app-building, product architecture",
        "model": os.getenv("AI_MODEL_NAME_OMNI", os.getenv("AI_MODEL_NAME", "facebook/opt-13b")),
        "tools": ["chat", "memory", "code_generation", "project_planning", "analysis", "multi_step_reasoning", "agent_execution", "appgrade"],
        "capabilities": ["chat", "memory", "code_generation", "analysis", "planning", "multi_step_reasoning", "agent_execution", "appgrade", "tool_use"],
    },
}

DEFAULT_TIER = os.getenv("AI_MODEL_TIER", "OMNI").upper()


def resolve_tier_config(tier_name: str | None = None) -> Dict[str, Any]:
    tier_key = (tier_name or DEFAULT_TIER or "EB").upper()
    if tier_key not in MODEL_TIERS:
        tier_key = "EB"
    return MODEL_TIERS[tier_key]


def get_runtime_context(persona_name: str | None = None, tier_name: str | None = None) -> Dict[str, Any]:
    tier = resolve_tier_config(tier_name)
    persona = (persona_name or "normal").lower()
    return {
        "identity": ENGINE_IDENTITY,
        "tier": tier,
        "persona": persona,
        "runtime": {
            "provider": "Transformers",
            "device": "cpu",
            "adapter_mode": "LoRA",
        },
    }


def get_experiment_config() -> Dict[str, Any]:
    """Return operational config for training jobs, checkpoints, and experiment tracking."""
    root_dir = ROOT_DIR
    return {
        "tracking_backend": os.getenv("TRACKING_BACKEND", "local").strip().lower(),
        "wandb_project": os.getenv("WANDB_PROJECT", "normal-engine").strip(),
        "wandb_entity": os.getenv("WANDB_ENTITY", "").strip(),
        "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "").strip(),
        "dvc_enabled": os.getenv("DVC_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"},
        "dataset_root": os.getenv("DATASET_ROOT", str(root_dir / "data" / "datasets")).strip(),
        "checkpoint_root": os.getenv("CHECKPOINT_ROOT", str(root_dir / "data" / "checkpoints")).strip(),
        "training_root": os.getenv("TRAINING_ROOT", str(root_dir / "data" / "training")).strip(),
        "model_registry_root": os.getenv("MODEL_REGISTRY_ROOT", str(root_dir / "data" / "models")).strip(),
        "log_dir": os.getenv("LOG_DIR", str(root_dir / "data" / "logs")).strip(),
    }


def get_security_config() -> Dict[str, Any]:
    """Return operational security and compliance settings."""
    return {
        "jwt_secret": os.getenv("API_JWT_SECRET") or os.getenv("SUPABASE_JWT_SECRET", "change-me").strip(),
        "rate_limit": int(os.getenv("API_RATE_LIMIT", "60")),
        "request_timeout_seconds": int(os.getenv("API_REQUEST_TIMEOUT_SECONDS", "30")),
        "allow_internal_ips": os.getenv("ALLOW_INTERNAL_IPS", "true").strip().lower() in {"1", "true", "yes", "on"},
        "billing_webhook_url": os.getenv("BILLING_WEBHOOK_URL", "").strip(),
        "compliance_mode": os.getenv("COMPLIANCE_MODE", "strict").strip().lower(),
    }


def get_model_registry() -> Dict[str, Any]:
    """Return the active model registry and rollout policy for generation tasks."""
    registry = {
        "image": {
            "default_model": os.getenv("AI_IMAGE_MODEL", "runwayml/stable-diffusion-v1-5"),
            "variants": [
                {
                    "name": "stable",
                    "model": os.getenv("AI_IMAGE_MODEL_STABLE", os.getenv("AI_IMAGE_MODEL", "runwayml/stable-diffusion-v1-5")),
                    "rollout": float(os.getenv("AI_IMAGE_MODEL_STABLE_ROLLOUT", "0.85")),
                    "enabled": os.getenv("AI_IMAGE_MODEL_STABLE_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"},
                },
                {
                    "name": "experimental",
                    "model": os.getenv("AI_IMAGE_MODEL_EXPERIMENTAL", os.getenv("AI_IMAGE_MODEL", "runwayml/stable-diffusion-v1-5")),
                    "rollout": float(os.getenv("AI_IMAGE_MODEL_EXPERIMENTAL_ROLLOUT", "0.15")),
                    "enabled": os.getenv("AI_IMAGE_MODEL_EXPERIMENTAL_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"},
                },
            ],
            "strategy": os.getenv("AI_IMAGE_ROLLOUT_STRATEGY", "weighted"),
        }
    }
    return registry


def resolve_model_variant(scope: str = "image", user_id: str | None = None, request_id: str | None = None) -> str:
    """Deterministically resolve the active model for a request using rollout weights."""
    registry = get_model_registry().get(scope, {})
    variants = registry.get("variants", [])
    if not variants:
        return registry.get("default_model", "")

    active_variants = [v for v in variants if v.get("enabled")]
    if not active_variants:
        return registry.get("default_model", "")

    context = f"{user_id or ''}:{request_id or ''}:{scope}"
    bucket = (sum((ord(ch) for ch in context)) % 100) / 100.0
    cumulative = 0.0
    selected = active_variants[0]

    for variant in active_variants:
        cumulative += float(variant.get("rollout", 0.0))
        if bucket <= cumulative:
            selected = variant
            break

    return str(selected.get("model") or registry.get("default_model", ""))


def get_music_config() -> Dict[str, Any]:
    """Return runtime configuration for symbolic music generation and raw-audio prototyping."""
    return {
        "backend": os.getenv("AI_MUSIC_BACKEND", "symbolic").strip().lower(),
        "model": os.getenv("AI_MUSIC_MODEL", "symbolic-transformer").strip(),
        "sample_rate": int(os.getenv("AI_MUSIC_SAMPLE_RATE", "22050")),
        "duration_seconds": float(os.getenv("AI_MUSIC_DURATION_SECONDS", "8.0")),
        "tempo_bpm": int(os.getenv("AI_MUSIC_TEMPO_BPM", "92")),
        "safety_mode": os.getenv("AI_MUSIC_SAFETY_MODE", "strict").strip().lower(),
    }


def get_video_config() -> Dict[str, Any]:
    """Return runtime configuration for low-resolution temporal video generation."""
    return {
        "backend": os.getenv("AI_VIDEO_BACKEND", "image-frame-diffusion").strip().lower(),
        "default_width": int(os.getenv("AI_VIDEO_WIDTH", "320")),
        "default_height": int(os.getenv("AI_VIDEO_HEIGHT", "180")),
        "default_frames": int(os.getenv("AI_VIDEO_FRAMES", "8")),
        "default_fps": int(os.getenv("AI_VIDEO_FPS", "8")),
        "motion_prior": os.getenv("AI_VIDEO_MOTION_PRIOR", "temporal-consistency").strip().lower(),
        "cache_enabled": os.getenv("AI_VIDEO_CACHE", "true").strip().lower() in {"1", "true", "yes", "on"},
        "cache_dir": os.getenv("AI_VIDEO_CACHE_DIR", str(ROOT_DIR / "data" / "generated_videos" / "cache")).strip(),
    }


def get_platform_config() -> Dict[str, Any]:
    """Return feature flags and operational config for the product platform layer."""
    return {
        "templates_enabled": os.getenv("PLATFORM_TEMPLATES_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"},
        "collaboration_enabled": os.getenv("PLATFORM_COLLABORATION_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"},
        "analytics_backend": os.getenv("ANALYTICS_BACKEND", "supabase").strip().lower(),
        "feedback_enabled": os.getenv("PLATFORM_FEEDBACK_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"},
        "monetization_enabled": os.getenv("PLATFORM_MONETIZATION_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"},
        "multilingual_enabled": os.getenv("PLATFORM_MULTILINGUAL_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"},
        "workspace_name": os.getenv("PLATFORM_WORKSPACE_NAME", "normal-platform").strip(),
    }


def get_quota_config() -> Dict[str, Any]:
    """Return usage and quota settings for premium generation tiers and platform access."""
    return {
        "default_quota_per_minute": int(os.getenv("API_QUOTA_PER_MINUTE", "60")),
        "default_quota_per_day": int(os.getenv("API_QUOTA_PER_DAY", "1000")),
        "pro_quota_per_minute": int(os.getenv("API_PRO_QUOTA_PER_MINUTE", "120")),
        "pro_quota_per_day": int(os.getenv("API_PRO_QUOTA_PER_DAY", "5000")),
        "premium_tiers": [tier.strip() for tier in os.getenv("PREMIUM_TIERS", "pro,premium,enterprise").split(",") if tier.strip()],
        "billing_required": os.getenv("BILLING_REQUIRED", "false").strip().lower() in {"1", "true", "yes", "on"},
    }


def list_model_registry(scope: str = "image") -> Dict[str, Any]:
    """Return the registry catalog and rollout metadata for a model family."""
    registry = get_model_registry().get(scope, {})
    defaults = {
        "scope": scope,
        "default_model": registry.get("default_model", ""),
        "strategy": registry.get("strategy", "weighted"),
        "variants": registry.get("variants", []),
    }
    return defaults


def get_generation_presets() -> Dict[str, Dict[str, Any]]:
    """Return built-in generation presets that can be consumed by backend workers and clients."""
    return {
        "cinematic": {
            "width": 1024,
            "height": 1024,
            "steps": 30,
            "guidance": 7.5,
            "temporal_consistency": True,
            "motion_prior": "cinematic",
        },
        "product-shot": {
            "width": 1024,
            "height": 1024,
            "steps": 26,
            "guidance": 8.0,
            "temporal_consistency": False,
            "motion_prior": "product",
        },
        "ambient-loop": {
            "width": 768,
            "height": 768,
            "steps": 25,
            "guidance": 7.0,
            "temporal_consistency": True,
            "motion_prior": "ambient",
        },
        "social-short": {
            "width": 720,
            "height": 1280,
            "steps": 28,
            "guidance": 7.5,
            "temporal_consistency": True,
            "motion_prior": "social-first",
        },
    }


def get_data_engineering_config() -> Dict[str, Any]:
    """Return storage, versioning, labeling, and preprocessing configuration for dataset operations."""
    return {
        "storage_backend": os.getenv("DATA_STORAGE_BACKEND", "s3").strip().lower(),
        "s3_endpoint": os.getenv("SUPABASE_S3_ENDPOINT", "").strip(),
        "s3_bucket": os.getenv("SUPABASE_STORAGE_BUCKET", "normal-assets").strip(),
        "dvc_enabled": os.getenv("DVC_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"},
        "versioning_backend": os.getenv("DATA_VERSIONING_BACKEND", "dvc").strip().lower(),
        "labeling_tool": os.getenv("LABELING_TOOL", "label-studio").strip().lower(),
        "preprocessing_audio": os.getenv("PREPROCESS_AUDIO", "ffmpeg+torchaudio").strip().lower(),
        "preprocessing_image": os.getenv("PREPROCESS_IMAGE", "albumentations").strip().lower(),
        "minio_enabled": os.getenv("MINIO_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"},
        "lifecycle_policies": os.getenv("LIFECYCLE_POLICIES", "archive,delete-old-versions").strip().lower(),
    }


def get_mlop_config() -> Dict[str, Any]:
    """Return deployment, orchestration, and observability settings for the model platform."""
    return {
        "orchestration": os.getenv("INFRA_ORCHESTRATION", "kubernetes").strip().lower(),
        "container_runtime": os.getenv("CONTAINER_RUNTIME", "docker").strip().lower(),
        "helm_enabled": os.getenv("HELM_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"},
        "experiment_tracking": os.getenv("TRACKING_BACKEND", "wandb").strip().lower(),
        "ci_cd": os.getenv("CI_CD_TOOL", "github-actions").strip().lower(),
        "monitoring": os.getenv("MONITORING_STACK", "prometheus+grafana").strip().lower(),
        "errors_backend": os.getenv("ERROR_BACKEND", "sentry").strip().lower(),
        "inference_backend": os.getenv("INFERENCE_BACKEND", "fastapi").strip().lower(),
        "streaming_backend": os.getenv("STREAMING_BACKEND", "webrtc").strip().lower(),
        "edge_runtime": os.getenv("EDGE_RUNTIME", "onnx").strip().lower(),
    }


def get_ethics_config() -> Dict[str, Any]:
    """Return required compliance and governance settings for production media generation."""
    return {
        "license_audit_required": os.getenv("LICENSE_AUDIT_REQUIRED", "true").strip().lower() in {"1", "true", "yes", "on"},
        "copyright_detection_enabled": os.getenv("COPYRIGHT_DETECTION_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"},
        "content_moderation_enabled": os.getenv("CONTENT_MODERATION_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"},
        "watermarking_enabled": os.getenv("WATERMARKING_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"},
        "user_consent_required": os.getenv("USER_CONSENT_REQUIRED", "true").strip().lower() in {"1", "true", "yes", "on"},
        "bias_testing_enabled": os.getenv("BIAS_TESTING_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"},
        "tos_required": os.getenv("TOS_REQUIRED", "true").strip().lower() in {"1", "true", "yes", "on"},
    }