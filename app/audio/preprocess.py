"""Reference-audio preprocessing: decode, trim, denoise, loudness-normalize.

XTTS v2 clones best from clean mono speech. We standardize everything to
24 kHz mono WAV, strip leading/trailing silence, suppress stationary noise,
and normalize integrated loudness to -23 LUFS (EBU R128-ish) with a -1 dBFS
peak guard.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import noisereduce as nr
import numpy as np
import pyloudnorm as pyln
import soundfile as sf

from app import config


@dataclass
class ProcessedReference:
    path: Path
    voice_id: str
    duration: float
    sample_rate: int
    warnings: list[str] = field(default_factory=list)


def _decode_with_ffmpeg(src: Path) -> tuple[np.ndarray, int]:
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(src),
        "-ac", "1", "-ar", str(config.SAMPLE_RATE),
        "-f", "f32le", "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    audio = np.frombuffer(raw, dtype=np.float32).copy()
    return audio, config.SAMPLE_RATE


def _decode(src: Path) -> tuple[np.ndarray, int]:
    try:
        y, sr = sf.read(str(src), dtype="float32", always_2d=True)
        y = y.mean(axis=1)
        if sr != config.SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=config.SAMPLE_RATE, res_type="soxr_hq")
        return np.ascontiguousarray(y, dtype=np.float32), config.SAMPLE_RATE
    except Exception:
        return _decode_with_ffmpeg(src)


def voice_id_for(path: Path) -> str:
    digest = hashlib.sha1(path.read_bytes()).hexdigest()
    return digest[:12]


def preprocess_reference(src: Path, out_dir: Path | None = None) -> ProcessedReference:
    out_dir = out_dir or config.REFERENCES_DIR
    warnings: list[str] = []

    y, sr = _decode(Path(src))
    duration = len(y) / sr

    if duration < config.MIN_REF_SECONDS:
        raise ValueError(
            f"Reference too short ({duration:.1f}s). Need at least {config.MIN_REF_SECONDS:.0f}s of speech."
        )
    if duration > config.MAX_REF_SECONDS:
        warnings.append(f"Trimmed from {duration:.0f}s to first {config.MAX_REF_SECONDS:.0f}s.")
        y = y[: int(config.MAX_REF_SECONDS * sr)]
        duration = len(y) / sr
    elif duration > config.WARN_REF_SECONDS:
        warnings.append(
            f"Reference is {duration:.0f}s; 20-30s of clean speech is ideal. Extra length adds latency without quality gain."
        )

    y = librosa.effects.trim(y, top_db=40)[0]
    if len(y) < sr * 1.0:
        warnings.append("Almost all silence detected after trimming; recording may be too quiet.")

    y = nr.reduce_noise(y=y, sr=sr, stationary=True, prop_decrease=0.85)

    meter = pyln.Meter(sr)
    try:
        loudness = meter.integrated_loudness(y.astype(np.float64))
        if np.isfinite(loudness):
            y = pyln.normalize.loudness(y.astype(np.float64), loudness, config.TARGET_LUFS)
            y = y.astype(np.float32)
    except Exception:
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak * 10 ** (-6 / 20)
        warnings.append("Loudness measurement failed; fell back to peak normalization.")

    peak = np.max(np.abs(y))
    if peak > 1.0:
        y = y / peak * (10 ** (-1.0 / 20))

    vid = voice_id_for(Path(src)) + f"-{int(duration)}"
    out_path = out_dir / f"{vid}.wav"
    sf.write(str(out_path), y, sr, subtype="PCM_16")

    return ProcessedReference(
        path=out_path, voice_id=vid, duration=len(y) / sr, sample_rate=sr, warnings=warnings,
    )


def probe_duration(src: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(src)],
        capture_output=True, check=True,
    ).stdout.decode().strip()
    m = re.match(r"\d+(\.\d+)?", out)
    return float(m.group(0)) if m else 0.0
