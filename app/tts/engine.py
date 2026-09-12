"""TTS engine registry: XTTS v2 (17 langs, fallback phonemizer) + OmniVoice
(646 langs incl. native Filipino `fil`).

Both engines lazily load their models once and cache per-voice artifacts in
data/embeddings/ (.npz = XTTS latents, .pt = OmniVoice clone prompts).

Licenses: XTTS v2 checkpoint — Coqui Public Model License (non-commercial;
COQUI_TOS_AGREED=1 set here). OmniVoice weights — CC-BY-NC (code Apache-2.0).
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import re
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from app import config

os.environ.setdefault("COQUI_TOS_AGREED", "1")


class VoiceNotFoundError(KeyError):
    pass


class EngineUnavailableError(RuntimeError):
    pass


def split_sentences(text: str, max_len: int = 240) -> list[str]:
    parts = re.split(r"(?<=[.!?…:])\s+|\n+", text)
    out: list[str] = []
    buf = ""
    for part in [p.strip() for p in parts if p.strip()]:
        if len(buf) + len(part) + 1 <= max_len:
            buf = f"{buf} {part}".strip()
        else:
            if buf:
                out.append(buf)
            while len(part) > max_len:
                cut = part.rfind(" ", 0, max_len)
                cut = cut if cut > max_len // 2 else max_len
                out.append(part[:cut].strip())
                part = part[cut:].strip()
            buf = part
    if buf:
        out.append(buf)
    return out


def _resolve_device() -> str:
    if config.DEVICE == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return config.DEVICE


def _fade(audio: np.ndarray, sr: int, fade_ms: int = 25) -> np.ndarray:
    """Apply a short fade-in and fade-out to smooth edges."""
    n = int(sr * fade_ms / 1000)
    if n < 1 or len(audio) < 2 * n:
        return audio
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n, dtype=np.float32)))
    out = audio.copy()
    out[:n] *= ramp
    out[-n:] *= ramp[::-1]
    return out


def _make_gap(sr: int, gap_ms: int = 40) -> np.ndarray:
    """A short silence gap to sit between chunks."""
    n = int(sr * gap_ms / 1000)
    return np.zeros(n, dtype=np.float32)


def _finish(audio: np.ndarray, voice_id: str, sr: int, t0: float,
            n_chunks: int, out_path: Path | None) -> tuple[Path, dict]:
    import pyloudnorm as pyln

    # Loudness-normalize to broadcast standard (-23 LUFS)
    meter = pyln.Meter(sr)
    try:
        loudness = meter.integrated_loudness(audio)
        if np.isfinite(loudness) and loudness > -70:
            audio = pyln.normalize.loudness(audio, loudness, config.TARGET_LUFS)
    except Exception:
        pass

    # True-peak safety clamp at -1 dBFS
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    ceiling = 10 ** (-1.0 / 20)  # -1 dBFS ≈ 0.891
    if peak > ceiling:
        audio *= ceiling / peak

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = out_path or (config.OUTPUT_DIR / f"tts-{stamp}-{voice_id}.wav")
    sf.write(str(out_path), audio, sr, subtype="PCM_16")
    stats = {
        "seconds": round(len(audio) / sr, 2),
        "rtf": round((time.time() - t0) / max(len(audio) / sr, 1e-6), 3),
        "chunks": n_chunks,
    }
    return out_path, stats


class XttsEngine:
    name = "xtts"

    def __init__(self) -> None:
        self._model = None
        self._device = None
        self._lock = threading.Lock()
        self._latents: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    @property
    def device(self) -> str:
        self._ensure_loaded()
        return str(self._device)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import transformers.pytorch_utils as tpu

        if not hasattr(tpu, "isin_mps_friendly"):
            tpu.isin_mps_friendly = torch.isin

        from TTS.api import TTS

        device = _resolve_device()
        ensure_exclusive(self)
        wrapper = TTS(config.XTTS_MODEL)
        model = wrapper.synthesizer.tts_model
        model.to(device)
        self._model = model
        self._device = device

    def _store_latents(self, voice_id: str, gpt_cond_latent: torch.Tensor,
                       speaker_embedding: torch.Tensor) -> None:
        np.savez(
            config.EMBEDDINGS_DIR / f"{voice_id}.npz",
            gpt_cond_latent=gpt_cond_latent.cpu().numpy(),
            speaker_embedding=speaker_embedding.cpu().numpy(),
        )
        self._latents[voice_id] = (
            gpt_cond_latent.cpu().numpy(), speaker_embedding.cpu().numpy(),
        )

    def register_reference(self, wav_path: Path, voice_id: str,
                           transcript: str = "") -> str:
        self._ensure_loaded()
        with self._lock:
            gpt_cond_latent, speaker_embedding = self._model.get_conditioning_latents(
                audio_path=[str(wav_path)]
            )
            self._store_latents(voice_id, gpt_cond_latent, speaker_embedding)
        return voice_id

    def rebuild(self, voice_id: str,
                samples: list[tuple[Path, str]]) -> dict:
        """Recompute conditioning latents over every stored sample at once.

        XTTS accepts multiple reference files and averages them, which steadies
        the timbre; transcripts are unused by this engine."""
        self._ensure_loaded()
        with self._lock:
            gpt_cond_latent, speaker_embedding = self._model.get_conditioning_latents(
                audio_path=[str(p) for p, _ in samples]
            )
            self._store_latents(voice_id, gpt_cond_latent, speaker_embedding)
        return {"mode": "averaged", "used": len(samples), "dropped": 0}

    def _load_latents(self, voice_id: str) -> tuple[np.ndarray, np.ndarray]:
        cached = self._latents.get(voice_id)
        if cached:
            return cached
        path = config.EMBEDDINGS_DIR / f"{voice_id}.npz"
        if not path.exists():
            raise VoiceNotFoundError(voice_id)
        data = np.load(path)
        pair = (data["gpt_cond_latent"], data["speaker_embedding"])
        self._latents[voice_id] = pair
        return pair

    def synthesize(self, text: str, language: str, voice_id: str,
                   out_path: Path | None = None,
                   instruct: str | None = None) -> tuple[Path, dict]:
        gpt_cond_latent, speaker_embedding = self._load_latents(voice_id)
        self._ensure_loaded()

        to_tensor = lambda a, dims: torch.tensor(np.asarray(a)).unsqueeze(0) if len(a.shape) < dims else torch.tensor(np.asarray(a))  # noqa: E731
        gpt_t = to_tensor(gpt_cond_latent, 3).to(self.device)
        spk_t = to_tensor(speaker_embedding, 2).to(self.device)

        chunks = split_sentences(text)
        pieces: list[np.ndarray] = []
        t0 = time.time()
        with self._lock, torch.inference_mode():
            for chunk in chunks:
                wav = self._model.inference(
                    text=chunk,
                    language=language,
                    gpt_cond_latent=gpt_t,
                    speaker_embedding=spk_t,
                    temperature=config.TEMPERATURE,
                    top_p=config.TOP_P,
                    top_k=config.TOP_K,
                    repetition_penalty=config.REPETITION_PENALTY,
                    length_penalty=config.LENGTH_PENALTY,
                    enable_text_splitting=False,
                )["wav"]
                pieces.append(np.asarray(wav, dtype=np.float32))
                pieces.append(_make_gap(config.SAMPLE_RATE))

        audio = np.concatenate(pieces)
        # Gentle fade-in/out at the very start/end (5ms) to avoid click artifacts
        audio = _fade(audio, config.SAMPLE_RATE, fade_ms=5)
        out_path, stats = _finish(audio, voice_id, config.SAMPLE_RATE, t0, len(chunks), out_path)
        return out_path, stats

    def forget(self, voice_id: str) -> None:
        self._latents.pop(voice_id, None)

    def reset(self) -> None:
        with self._lock:
            self._model = None
            self._latents.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @property
    def sample_rate(self) -> int:
        return config.SAMPLE_RATE


class OmniEngine:
    """OmniVoice (k2-fsa): diffusion-LM zero-shot TTS, 646 languages incl.
    native Filipino (`fil`). Reference best practice: 3-10s clip."""

    name = "omni"

    def __init__(self) -> None:
        self._model = None
        self._device = None
        self._lock = threading.Lock()
        self._prompts: dict[str, object] = {}

    @property
    def device(self) -> str:
        self._ensure_loaded()
        return str(self._device)

    def _ensure_loaded(self):
        if self._model is not None:
            return self._model
        try:
            from omnivoice import OmniVoice
        except ImportError as exc:
            raise EngineUnavailableError(
                "omnivoice is not installed (pip install omnivoice)"
            ) from exc
        device = _resolve_device()
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        ensure_exclusive(self)
        model = OmniVoice.from_pretrained(
            config.OMNI_MODEL, device_map=device, dtype=dtype,
        )
        if device.startswith("cuda"):
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if vram_gb < 8:
                model.audio_tokenizer = model.audio_tokenizer.to("cpu")
        self._model = model
        self._device = device
        return model

    name = "omni"
    _TERMINAL_PUNCT = tuple(".!?…")

    @staticmethod
    def _pause_cuts(wav_path: Path, targets: list[float]) -> list[Path]:
        """Candidate prefixes ending at >=250ms pauses, one per target length.

        The clone prompt's ref_text+ref_audio are context the LM continues:
        if the transcript ends mid-clause, generation opens by finishing that
        stray clause before the requested text (heard as an extra voice at
        the start). Cutting at real pauses keeps the auto-transcribed
        reference sentence-complete."""
        y, sr = sf.read(str(wav_path), dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        dur = len(y) / sr
        hop_ms = 10
        hop = int(hop_ms / 1000 * sr)
        env = np.array([np.sqrt(np.mean(y[i:i + hop] ** 2))
                        for i in range(0, len(y) - hop, hop)])
        quiet = env < max(0.01, float(env.max()) * 0.05)
        min_gap = int(0.25 * 1000 / hop_ms)
        mids: list[float] = []
        run_start = None
        for i, q in enumerate(quiet):
            if q and run_start is None:
                run_start = i
            elif not q and run_start is not None:
                if i - run_start >= min_gap:
                    mid = ((run_start + i) // 2) * hop_ms / 1000
                    if 5.0 <= mid <= dur - 1.0:
                        mids.append(mid)
                run_start = None
        paths = []
        used: set[int] = set()
        for tgt in targets:
            if not mids:
                break
            n = int(min(mids, key=lambda m: abs(m - tgt)) * sr)
            if any(abs(n - u) < sr for u in used) or n < 2 * sr:
                continue
            used.add(n)
            out = config.TMP_DIR / f"{wav_path.stem}-cut{len(paths)}.wav"
            sf.write(str(out), y[:n], sr, subtype="PCM_16")
            paths.append(out)
        return paths

    def register_reference(self, wav_path: Path, voice_id: str,
                           transcript: str = "") -> str:
        model = self._ensure_loaded()
        text = (transcript or "").strip() or None

        cuts: list[Path] = []
        try:
            with self._lock:
                if text is not None:
                    if not text.endswith(self._TERMINAL_PUNCT):
                        print(f"[makata] warning: transcript for '{voice_id}' "
                              "does not end a sentence; output may parrot its "
                              "tail. Supply text matching the audio exactly.")
                    prompt = model.create_voice_clone_prompt(
                        ref_audio=str(wav_path), ref_text=text)
                else:
                    # No transcript: cut at real pauses near the target length
                    # and keep only cuts whose transcription ends complete.
                    prompt = None
                    cuts = self._pause_cuts(
                        wav_path, [config.OMNI_TARGET_REF_SECONDS, 8.0])
                    for cut in cuts:
                        prompt = model.create_voice_clone_prompt(
                            ref_audio=str(cut), ref_text=None)
                        if prompt.ref_text.endswith(self._TERMINAL_PUNCT):
                            break
                    if prompt is None:
                        prompt = model.create_voice_clone_prompt(
                            ref_audio=str(wav_path), ref_text=None)
                prompt.save(str(config.EMBEDDINGS_DIR / f"{voice_id}.pt"))
                self._prompts[voice_id] = prompt
        finally:
            for c in cuts:
                c.unlink(missing_ok=True)
        return voice_id

    @staticmethod
    def _clip_duration(wav_path: Path) -> float:
        info = sf.info(str(wav_path))
        return info.frames / max(info.samplerate, 1)

    def rebuild(self, voice_id: str,
                samples: list[tuple[Path, str]]) -> dict:
        """Rebuild the clone prompt from all stored samples.

        `samples` is [(wav_path, transcript)] in priority order. Transcript-
        bearing clips are stitched — audio concatenated at short silences,
        texts joined word-for-word in the same order — into a single prompt
        capped at OMNI_MAX_PROMPT_SECONDS (README advises 3-20s refs). Without
        any transcripts, the longest clip falls back to the Whisper
        auto-transcribe path of register_reference."""
        model = self._ensure_loaded()
        if not samples:
            raise VoiceNotFoundError(voice_id)
        spoken = [(p, t.strip()) for p, t in samples if (t or "").strip()]

        if not spoken:
            best = max(samples, key=lambda s: self._clip_duration(s[0]))[0]
            self.register_reference(best, voice_id)
            return {"mode": "single-auto", "used": 1, "dropped": len(samples) - 1}

        # Greedy prefix: keep clips in order while they fit the prompt cap.
        sel: list[tuple[Path, str]] = []
        total = 0.0
        for p, t in spoken:
            dur = self._clip_duration(p)
            if sel and total + config.SAMPLE_GAP_SECONDS + dur > config.OMNI_MAX_PROMPT_SECONDS:
                break
            sel.append((p, t))
            total += dur + (config.SAMPLE_GAP_SECONDS if len(sel) > 1 else 0.0)
        dropped = len(spoken) - len(sel)

        with self._lock:
            if len(sel) == 1:
                p, text = sel[0]
                prompt = model.create_voice_clone_prompt(ref_audio=str(p), ref_text=text)
            else:
                gap = np.zeros(int(config.SAMPLE_RATE * config.SAMPLE_GAP_SECONDS),
                               dtype=np.float32)
                pieces: list[np.ndarray] = []
                texts: list[str] = []
                sr = config.SAMPLE_RATE
                for i, (p, t) in enumerate(sel):
                    y, clip_sr = sf.read(str(p), dtype="float32")
                    if y.ndim > 1:
                        y = y.mean(axis=1)
                    sr = clip_sr
                    if i:
                        pieces.append(gap)
                    pieces.append(y)
                    texts.append(t)
                tmp = config.TMP_DIR / f"{voice_id}-stitched.wav"
                try:
                    sf.write(str(tmp), np.concatenate(pieces), sr, subtype="PCM_16")
                    prompt = model.create_voice_clone_prompt(
                        ref_audio=str(tmp), ref_text=" ".join(texts))
                finally:
                    tmp.unlink(missing_ok=True)
            if not prompt.ref_text.endswith(self._TERMINAL_PUNCT):
                print(f"[makata] warning: rebuilt prompt for '{voice_id}' does not "
                      "end a sentence; output may parrot its tail.")
            prompt.save(str(config.EMBEDDINGS_DIR / f"{voice_id}.pt"))
            self._prompts[voice_id] = prompt
        return {"mode": "stitched" if len(sel) > 1 else "single",
                "used": len(sel), "dropped": dropped}

    def synthesize(self, text: str, language: str, voice_id: str,
                   out_path: Path | None = None,
                   instruct: str | None = None) -> tuple[Path, dict]:
        model = self._ensure_loaded()
        prompt = self._prompts.get(voice_id)
        if prompt is None:
            pt = config.EMBEDDINGS_DIR / f"{voice_id}.pt"
            if not pt.exists():
                raise VoiceNotFoundError(voice_id)
            from omnivoice import VoiceClonePrompt

            prompt = VoiceClonePrompt.load(str(pt))
            self._prompts[voice_id] = prompt

        chunks = split_sentences(text)
        pieces: list[np.ndarray] = []
        t0 = time.time()
        with self._lock, torch.inference_mode():
            for chunk in chunks:
                gen_kwargs: dict = dict(
                    text=chunk,
                    voice_clone_prompt=prompt,
                    num_step=config.OMNI_NUM_STEP,
                )
                if instruct:
                    gen_kwargs["instruct"] = instruct
                wav = model.generate(**gen_kwargs)
                piece = np.asarray(wav[0], dtype=np.float32)
                pieces.append(piece)
                pieces.append(_make_gap(model.sampling_rate))

        audio = np.concatenate(pieces)
        # Gentle fade-in/out at the very start/end (5ms) to avoid click artifacts
        audio = _fade(audio, model.sampling_rate, fade_ms=5)
        out_path, stats = _finish(audio, voice_id, model.sampling_rate, t0, len(chunks), out_path)
        return out_path, stats

    def forget(self, voice_id: str) -> None:
        self._prompts.pop(voice_id, None)

    def reset(self) -> None:
        with self._lock:
            self._model = None
            self._prompts.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @property
    def sample_rate(self) -> int:
        if self._model is not None:
            return int(self._model.sampling_rate)
        return config.SAMPLE_RATE


_xtts_engine = XttsEngine()
_omni_engine = OmniEngine()

_ALL_ENGINES = {"xtts": _xtts_engine, "omni": _omni_engine}


def ensure_exclusive(current) -> None:
    """4GB-class GPUs cannot hold both models; unload the other engine."""
    for eng in _ALL_ENGINES.values():
        if eng is not current and eng._model is not None:
            eng.reset()


def get_engine(name: str):
    name = (name or "").strip().lower()
    if name == "xtts":
        return _xtts_engine
    if name == "omni":
        return _omni_engine
    raise ValueError(f"Unknown engine '{name}'")


def engine_for_voice(voice_id: str) -> str:
    """Infer owning engine from stored artifact extension."""
    if (config.EMBEDDINGS_DIR / f"{voice_id}.pt").exists():
        return "omni"
    if (config.EMBEDDINGS_DIR / f"{voice_id}.npz").exists():
        return "xtts"
    raise VoiceNotFoundError(voice_id)


engine = get_engine(config.DEFAULT_ENGINE)
