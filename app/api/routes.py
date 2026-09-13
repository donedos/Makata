"""FastAPI routes: reference upload, synthesis, voice management."""
from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app import config
from app.audio.preprocess import preprocess_reference
from app.text.drill import drill_pack
from app.text.normalizer import detect_language, normalize_text
from app.tts.engine import (
    EngineUnavailableError,
    VoiceNotFoundError,
    engine_for_voice,
    get_engine,
)

router = APIRouter(prefix="/api")

_LANG_ALIASES = {"tl", "fil", "tagalog", "filipino"}

VOICES_META = config.DATA_DIR / "voices.json"
OUTPUTS_META = config.DATA_DIR / "output_history.json"


def _load_meta() -> dict:
    if VOICES_META.exists():
        try:
            return json.loads(VOICES_META.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_meta(meta: dict) -> None:
    VOICES_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _clean_name(name: str, fallback: str) -> str:
    return ((name or "").strip() or fallback)[:64]


def _load_out() -> dict:
    if OUTPUTS_META.exists():
        try:
            return json.loads(OUTPUTS_META.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_out(meta: dict) -> None:
    OUTPUTS_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_language(requested: str, text: str) -> tuple[str, str]:
    """Return (normalize_lang, native_lang). normalize_lang is 'tl' or 'en'
    for word expansion; native_lang is the actual language being spoken
    ('fil' for Tagalog) — engines map it to their own codes downstream."""
    lang = requested.strip().lower()
    if not lang or lang == "auto":
        detected = detect_language(text)
        return detected, detected
    if lang in _LANG_ALIASES:
        return "tl", "fil"
    base = "en" if lang.startswith("zh") else lang
    return base, base


class SynthesizeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    voice_id: str
    language: str = "auto"
    instruct: str | None = None


class RenameRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64)


def _voice_engine(voice_id: str) -> str:
    meta = _load_meta()
    bound = meta.get(voice_id, {}).get("engine")
    if bound in config.ENGINES:
        return bound
    return engine_for_voice(voice_id)


def _samples_dir(voice_id: str) -> Path:
    return config.REFERENCES_DIR / voice_id


def _sample_entries(meta: dict, voice_id: str) -> list[dict]:
    """Stored samples [(file-relative-to-REFERENCES_DIR, transcript)].

    Legacy voices registered before multi-sample support have no 'samples'
    key; fall back to their single primary reference with an unknown
    (empty) transcript."""
    entry = meta.get(voice_id) or {}
    samples = entry.get("samples")
    if samples is None:
        primary = config.REFERENCES_DIR / f"{voice_id}.wav"
        samples = [{"file": f"{voice_id}.wav", "transcript": ""}] if primary.exists() else []
    return [s for s in samples if (config.REFERENCES_DIR / s["file"]).exists()]


def _resolved_samples(samples: list[dict]) -> list[tuple[Path, str]]:
    return [(config.REFERENCES_DIR / s["file"], s.get("transcript", "")) for s in samples]


@router.post("/upload-reference")
async def upload_reference(
    file: UploadFile = File(...),
    name: str = Form(""),
    transcript: str = Form(""),
):
    engine_name = config.DEFAULT_ENGINE
    suffix = Path(file.filename or "ref.wav").suffix.lower() or ".wav"
    tmp_in = config.TMP_DIR / f"{uuid.uuid4().hex}{suffix}"
    with tmp_in.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)
    try:
        processed = preprocess_reference(tmp_in)
    except ValueError as exc:
        tmp_in.unlink(missing_ok=True)
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        tmp_in.unlink(missing_ok=True)
        raise HTTPException(400, f"Could not decode audio: {exc}") from exc
    finally:
        file.file.close()

    try:
        get_engine(engine_name).register_reference(
            processed.path, processed.voice_id, transcript=transcript,
        )
    except EngineUnavailableError as exc:
        raise HTTPException(501, str(exc)) from exc
    display = _clean_name(name, processed.voice_id)
    meta = _load_meta()
    meta[processed.voice_id] = {
        "name": display,
        "engine": engine_name,
        "samples": [{"file": f"{processed.voice_id}.wav",
                     "transcript": (transcript or "").strip()}],
    }
    _save_meta(meta)
    return {
        "voice_id": processed.voice_id,
        "name": display,
        "engine": engine_name,
        "duration": round(processed.duration, 2),
        "sample_rate": processed.sample_rate,
        "warnings": processed.warnings,
    }


@router.post("/synthesize")
async def synthesize(req: SynthesizeRequest):
    norm_lang, native_lang = _resolve_language(req.language, req.text)
    normalized = normalize_text(req.text, lang=norm_lang)
    try:
        engine_name = _voice_engine(req.voice_id)
        eng = get_engine(engine_name)
        if engine_name == "omni":
            tts_lang = "fil" if native_lang == "tl" else native_lang
        else:
            tts_lang = config.TAGALOG_FALLBACK_LANG if native_lang == "fil" else native_lang
        out, stats = eng.synthesize(normalized, tts_lang, req.voice_id,
                                    instruct=req.instruct)
    except VoiceNotFoundError as exc:
        raise HTTPException(404, f"Unknown voice_id '{req.voice_id}'. Upload a reference first.") from exc
    except EngineUnavailableError as exc:
        raise HTTPException(501, str(exc)) from exc
    except RuntimeError as exc:
        detail = str(exc)
        if "out of memory" in detail.lower():
            get_engine(engine_name).reset()
            raise HTTPException(507, "GPU out of memory; model reset. Retry or set MAKATA_DEVICE=cpu.") from exc
        raise HTTPException(500, detail) from exc
    out_name = Path(out).name
    meta = _load_out()
    meta[out_name] = {
        "text": req.text,
        "voice_id": req.voice_id,
        "voice_name": _load_meta().get(req.voice_id, {}).get("name"),
        "engine": engine_name,
        "language_used": tts_lang,
        "seconds": stats.get("seconds"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    _save_out(meta)
    return {
        "audio_url": f"/audio/{out_name}",
        "engine": engine_name,
        "language_used": tts_lang,
        "normalized_text": normalized,
        **stats,
    }


@router.get("/outputs")
async def list_outputs():
    meta = _load_out()
    items = []
    for p in config.OUTPUT_DIR.glob("*.wav"):
        st = p.stat()
        m = meta.get(p.name, {})
        items.append({
            "file": p.name,
            "audio_url": f"/audio/{p.name}",
            "text": m.get("text"),
            "voice_id": m.get("voice_id"),
            "voice_name": m.get("voice_name"),
            "language_used": m.get("language_used"),
            "seconds": m.get("seconds", round(st.st_size / (config.SAMPLE_RATE * 2), 1)),
            "created_at": m.get("created_at")
            or datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
        })
    items.sort(key=lambda x: x["created_at"], reverse=True)
    return items


@router.delete("/outputs/{filename}")
async def delete_output(filename: str):
    safe = Path(filename).name
    if not safe.endswith(".wav"):
        raise HTTPException(400, "Invalid filename")
    p = config.OUTPUT_DIR / safe
    if not p.exists():
        raise HTTPException(404, "Output not found")
    p.unlink()
    meta = _load_out()
    meta.pop(safe, None)
    _save_out(meta)
    return {"deleted": safe}


@router.get("/voices")
async def list_voices():
    meta = _load_meta()
    voices = sorted(config.EMBEDDINGS_DIR.glob("*.npz")) + sorted(config.EMBEDDINGS_DIR.glob("*.pt"))
    out = []
    for p in voices:
        vid = p.stem
        entry = meta.get(vid, {})
        try:
            eng = _voice_engine(vid)
        except VoiceNotFoundError:
            continue
        out.append({
            "voice_id": vid,
            "name": entry.get("name", vid),
            "engine": entry.get("engine", eng),
            "samples": len(_sample_entries(meta, vid)),
        })
    return sorted(out, key=lambda v: v["name"].lower())


@router.patch("/voices/{voice_id}")
async def rename_voice(voice_id: str, req: RenameRequest):
    try:
        engine_for_voice(voice_id)
    except VoiceNotFoundError as exc:
        raise HTTPException(404, f"Unknown voice_id '{voice_id}'") from exc
    clean = req.name.strip()[:64]
    if not clean:
        raise HTTPException(422, "Name cannot be empty or whitespace")
    meta = _load_meta()
    entry = meta.get(voice_id, {})
    entry["name"] = clean
    meta[voice_id] = entry
    _save_meta(meta)
    return {"voice_id": voice_id, "name": clean}


@router.get("/voices/{voice_id}/reference")
async def voice_reference(voice_id: str):
    p = config.REFERENCES_DIR / f"{voice_id}.wav"
    if not p.exists():
        raise HTTPException(404, "Reference clip not found for this voice")
    return FileResponse(p, media_type="audio/wav", filename=f"{voice_id}.wav")


@router.post("/voices/{voice_id}/samples")
async def add_voice_sample(
    voice_id: str,
    file: UploadFile = File(...),
    transcript: str = Form(""),
):
    """Append an extra reference clip (ideally with its exact transcript) and
    rebuild the clone prompt from all stored samples."""
    try:
        engine_name = _voice_engine(voice_id)
    except VoiceNotFoundError as exc:
        raise HTTPException(404, f"Unknown voice_id '{voice_id}'") from exc

    suffix = Path(file.filename or "sample.wav").suffix.lower() or ".wav"
    tmp_in = config.TMP_DIR / f"{uuid.uuid4().hex}{suffix}"
    with tmp_in.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)
    try:
        processed = preprocess_reference(tmp_in, out_dir=config.TMP_DIR)
    except ValueError as exc:
        tmp_in.unlink(missing_ok=True)
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        tmp_in.unlink(missing_ok=True)
        raise HTTPException(400, f"Could not decode audio: {exc}") from exc
    finally:
        file.file.close()

    tmp_in.unlink(missing_ok=True)

    out_dir = _samples_dir(voice_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("sample-*.wav"))
    nxt = int(existing[-1].stem.rsplit("-", 1)[1]) + 1 if existing else 0
    dest_rel = f"{voice_id}/sample-{nxt:03d}.wav"
    dest_abs = config.REFERENCES_DIR / dest_rel
    shutil.move(str(processed.path), str(dest_abs))

    meta = _load_meta()
    entry = meta.get(voice_id) or {}
    samples = _sample_entries(meta, voice_id)
    samples.append({"file": dest_rel, "transcript": (transcript or "").strip()})
    entry["samples"] = samples

    warnings = list(processed.warnings)
    clean_transcript = (transcript or "").strip()
    if engine_name == "omni" and not clean_transcript:
        warnings.append(
            "Sample stored without a transcript — OmniVoice stitches only "
            "transcript-backed clips; with none at all it falls back to a "
            "single auto-transcribed clip.")
    try:
        rebuild = get_engine(engine_name).rebuild(voice_id, _resolved_samples(samples))
    except EngineUnavailableError as exc:
        dest_abs.unlink(missing_ok=True)
        raise HTTPException(501, str(exc)) from exc

    meta[voice_id] = entry
    _save_meta(meta)
    return {
        "voice_id": voice_id,
        "name": entry.get("name", voice_id),
        "engine": engine_name,
        "samples": len(samples),
        "rebuild": rebuild,
        "duration": round(processed.duration, 2),
        "sample_rate": processed.sample_rate,
        "warnings": warnings,
    }


@router.get("/voices/{voice_id}/samples")
async def list_voice_samples(voice_id: str):
    try:
        _voice_engine(voice_id)
    except VoiceNotFoundError as exc:
        raise HTTPException(404, f"Unknown voice_id '{voice_id}'") from exc
    meta = _load_meta()
    items = []
    for i, s in enumerate(_sample_entries(meta, voice_id)):
        p = config.REFERENCES_DIR / s["file"]
        items.append({
            "index": i,
            "transcript": s.get("transcript") or None,
            "audio_url": f"/api/voices/{voice_id}/samples/{i}",
            "duration": round(p.stat().st_size / 2 / config.SAMPLE_RATE, 2),
        })
    return {"voice_id": voice_id, "samples": items}


@router.get("/voices/{voice_id}/samples/{index}")
async def voice_sample_audio(voice_id: str, index: int):
    meta = _load_meta()
    samples = _sample_entries(meta, voice_id)
    if index < 0 or index >= len(samples):
        raise HTTPException(404, f"Sample {index} not found for this voice")
    p = config.REFERENCES_DIR / samples[index]["file"]
    if not p.exists():
        raise HTTPException(404, "Sample file missing")
    return FileResponse(p, media_type="audio/wav", filename=p.name)


@router.delete("/voices/{voice_id}")
async def delete_voice(voice_id: str):
    artifacts = [
        p for p in (config.EMBEDDINGS_DIR / f"{voice_id}.npz",
                    config.EMBEDDINGS_DIR / f"{voice_id}.pt")
        if p.exists()
    ]
    if not artifacts:
        raise HTTPException(404, f"Unknown voice_id '{voice_id}'")
    for p in artifacts:
        p.unlink()
    (config.REFERENCES_DIR / f"{voice_id}.wav").unlink(missing_ok=True)
    shutil.rmtree(config.REFERENCES_DIR / voice_id, ignore_errors=True)
    get_engine("xtts").forget(voice_id)
    try:
        get_engine("omni").forget(voice_id)
    except Exception:
        pass
    meta = _load_meta()
    meta.pop(voice_id, None)
    _save_meta(meta)
    return {"deleted": voice_id}


@router.get("/health")
async def health():
    try:
        import omnivoice  # noqa: F401
        omni_available = True
    except ImportError:
        omni_available = False
    return {
        "status": "ok",
        "default_engine": config.DEFAULT_ENGINE,
        "engines": {"omni": {"available": omni_available}, "xtts": {"available": True}},
        "device": ("cuda" if __import__("torch").cuda.is_available() else "cpu"),
    }


@router.get("/languages")
async def languages():
    return {
        "default_engine": config.DEFAULT_ENGINE,
        "xtts_languages": config.XTTS_LANGUAGES,
        "tagalog_note": "OmniVoice synthesizes Filipino (fil) natively — no fallback needed. "
                        "XTTS v2 lacks Tagalog and routes tl text through the "
                        f"'{config.TAGALOG_FALLBACK_LANG}' phonemizer "
                        "(es = Spanish, orthography-aligned with Filipino; en is "
                        "an alternative).",
        "tagalog_fallback": config.TAGALOG_FALLBACK_LANG,
    }


@router.get("/drill-pack")
async def drill_pack_read(count: int = 5):
    """Pronunciation-drill sentences for sharpening a voice's clone prompt.

    Each item is one short sentence to record and add as an extra sample via
    POST /api/voices/{voice_id}/samples with the matching transcript. Ordered
    by priority; `count` caps the number returned (default 5)."""
    return {"pack": drill_pack(count), "hint": "Record each line aloud, then "
            "add it as a sample with its exact transcript."}
