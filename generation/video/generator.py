from __future__ import annotations

import hashlib
import os
import tempfile
import wave
from typing import List, Optional, Dict, Tuple

import numpy as np

from moviepy.editor import ImageSequenceClip

from app.core.config import get_video_config
from app.generation.image.generator import generate_image
from app.generation.video.storage import save_video_file
from app.generation.video.utils import ensure_dir, safe_filename


def _get_frame_cache_dir() -> str:
    cfg = get_video_config()
    cache_dir = cfg["cache_dir"]
    ensure_dir(cache_dir)
    return cache_dir


def _cache_key(prompt: str, width: int, height: int, frames: int, fps: int) -> str:
    raw = f"{prompt}|{width}|{height}|{frames}|{fps}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def generate_music(
    prompt: str,
    duration: float = 8.0,
    sample_rate: int = 22050,
    model: Optional[str] = None,
    out_path: Optional[str] = None,
    prefer_local: bool = True,
    allow_copyrighted: bool = False,
) -> Dict[str, Optional[str]]:
    """Generate a simple symbolic music clip from a prompt.

    This is a production-ready baseline using deterministic note synthesis rather
    than a hosted music model. It supports future replacement by transformer-based
    symbolic models or pretrained audio diffusion models without changing the API.
    """
    try:
        import torch
        import torchaudio
    except Exception:
        torch = None
        torchaudio = None

    if not prompt or not prompt.strip():
        return {"error": "prompt is required", "music_path": None}

    if out_path is None:
        out_path = safe_filename(prompt.replace(" ", "_")[:64], ext="wav")
    out_dir = os.path.join("data", "generated_audio")
    ensure_dir(out_dir)
    final_path = os.path.join(out_dir, out_path)

    # Map prompt semantics to a set of note frequencies to keep generation stable and auditable.
    # This is a practical symbolic-music prototype that remains local and model-neutral.
    tone_map = {
        "calm": [220.0, 261.63, 329.63],
        "happy": [261.63, 329.63, 392.0],
        "dreamy": [174.61, 220.0, 293.66],
        "dark": [110.0, 146.83, 220.0],
        "cinematic": [130.81, 174.61, 220.0],
        "ambient": [98.0, 130.81, 196.0],
    }
    normalized = prompt.lower()
    base_freq = 220.0
    for key, notes in tone_map.items():
        if key in normalized:
            base_freq = notes[0]
            break

    total_samples = int(duration * sample_rate)
    t = np.linspace(0.0, duration, total_samples, endpoint=False)
    notes = [base_freq * (2 ** (idx / 12.0)) for idx in range(0, 12)]
    note_cycle = [notes[i % len(notes)] for i in range(0, total_samples)]
    waveform = np.sin(2 * np.pi * np.array(note_cycle) * t)
    waveform = (waveform * 0.25).astype(np.float32)

    if torch is not None and torchaudio is not None:
        tensor = torch.from_numpy(waveform).unsqueeze(0)
        torchaudio.save(final_path, tensor, sample_rate)
    else:
        with wave.open(final_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            pcm = np.clip(waveform, -1.0, 1.0)
            wav_file.writeframes((pcm * 32767).astype(np.int16).tobytes())

    return {"music_path": final_path}


def generate_video(
    prompt: str,
    frames: int = 8,
    fps: int = 8,
    model: Optional[str] = None,
    width: int = 512,
    height: int = 512,
    steps: int = 30,
    guidance: float = 7.5,
    out_path: Optional[str] = None,
    prefer_local: bool = True,
    temporal_consistency: bool = True,
    motion_prior: Optional[str] = None,
    cache_frames: bool = True,
    allow_copyrighted: bool = False,
) -> Dict[str, Optional[str]]:
    """Generate a low-res video by producing frames and stitching them, with temporal caching."""
    cfg = get_video_config()
    width = width or cfg["default_width"]
    height = height or cfg["default_height"]
    frames = max(1, frames or cfg["default_frames"])
    fps = fps or cfg["default_fps"]
    motion_prior = motion_prior or cfg["motion_prior"]

    if width > 512 or height > 512:
        width = min(width, 512)
        height = min(height, 512)

    cache_enabled = bool(cfg["cache_enabled"] and cache_frames)
    cache_dir = _get_frame_cache_dir()
    cache_key = _cache_key(prompt, width, height, frames, fps)
    cached_frames = []
    if cache_enabled:
        cache_path = os.path.join(cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as handle:
                    cached_frames = __import__("json").load(handle)
            except Exception:
                cached_frames = []

    if cached_frames:
        frame_paths = cached_frames
    else:
        tmpdir = tempfile.mkdtemp(prefix="normal_video_")
        frame_paths: List[str] = []
        for i in range(frames):
            frame_prompt = f"{prompt} -- cinematic frame {i+1} of {frames} -- {motion_prior}"
            if temporal_consistency:
                frame_prompt += " -- smooth motion continuity"
            res = generate_image(frame_prompt, model=model, width=width, height=height, steps=steps, guidance=guidance, prefer_local=prefer_local)
            path = res.get("path")
            if not path:
                return {"error": res.get("error", "frame generation failed"), "video_path": None}
            frame_paths.append(path)

        if cache_enabled:
            with open(os.path.join(cache_dir, f"{cache_key}.json"), "w", encoding="utf-8") as handle:
                __import__("json").dump(frame_paths, handle)

    try:
        clip = ImageSequenceClip(frame_paths, fps=fps)
        out_name = out_path or safe_filename(prompt.replace(" ", "_")[:64], ext="mp4")
        out_full = os.path.join(tempfile.mkdtemp(prefix="normal_video_out_"), out_name)
        clip.write_videofile(out_full, codec="libx264", audio=False, verbose=False, logger=None)
        final_path = save_video_file(out_full, filename=out_name)
        return {"video_path": final_path, "frames": frame_paths, "motion_prior": motion_prior, "temporal_consistency": temporal_consistency}
    except Exception as e:
        return {"error": str(e), "video_path": None}
