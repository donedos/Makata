"""Makata command-line interface.

Usage:
    makata serve [--host H] [--port P] [--engine omni|xtts] [--device cuda|cpu]
    makata check-setup [--model]
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys


def _apply_serve_args(args: argparse.Namespace) -> None:
    if args.engine:
        os.environ["MAKATA_ENGINE"] = args.engine
    if args.device:
        os.environ["MAKATA_DEVICE"] = args.device
    if args.data_dir:
        os.environ["MAKATA_DATA_DIR"] = str(args.data_dir)


def cmd_serve(args: argparse.Namespace) -> int:
    _apply_serve_args(args)
    try:
        import uvicorn
    except ImportError as e:
        print(f"Makata is not fully installed ({e}).\nRun:  pip install makata   "
              "(or pip install -e . from a checkout)")
        return 1
    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def cmd_check_setup(args: argparse.Namespace) -> int:
    checks: list[tuple[str, bool, str]] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))
        mark = "OK  " if ok else "FAIL"
        print(f"[{mark}] {name}{(' — ' + detail) if detail else ''}")

    print(f"python      : {sys.version.split()[0]}")

    try:
        import torch
        add("torch", True, torch.__version__)
        add("cuda", torch.cuda.is_available(),
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "not available (CPU mode)")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
        if vram and vram < 6:
            print(f"       note: {vram:.1f} GB VRAM is below the 8 GB recommendation; "
                  "set MAKATA_DEVICE=cpu if you hit out-of-memory")
    except ImportError as e:
        add("torch", False, str(e))

    add("ffmpeg", shutil.which("ffmpeg") is not None,
        shutil.which("ffmpeg") or "install ffmpeg for broad audio-format support")

    for mod in ("omnivoice", "fastapi", "librosa", "noisereduce", "pyloudnorm", "soundfile"):
        try:
            __import__(mod)
            add(f"dep:{mod}", True)
        except ImportError as e:
            add(f"dep:{mod}", False, str(e))

    try:
        import torch as _t
        import transformers.pytorch_utils as _tpu

        if not hasattr(_tpu, "isin_mps_friendly"):
            _tpu.isin_mps_friendly = _t.isin
        __import__("TTS")
        add("dep:TTS (xtts)", True)
    except Exception as e:
        add("dep:TTS (xtts)", False, str(e))

    from app import config
    config.ensure_dirs()
    add("data dirs", all(d.exists() for d in (
        config.DATA_DIR, config.REFERENCES_DIR, config.EMBEDDINGS_DIR, config.OUTPUT_DIR)))
    print(f"       data dir: {config.DATA_DIR}")

    print(f"\nDefault engine: {config.DEFAULT_ENGINE}")
    print(f"XTTS model    : {config.XTTS_MODEL} (tl -> '{config.TAGALOG_FALLBACK_LANG}' phonemizer)")
    print(f"OmniVoice     : {config.OMNI_MODEL} (native fil, num_step={config.OMNI_NUM_STEP})")
    print("licenses      : XTTS v2 = CPML (non-commercial); OmniVoice weights = CC-BY-NC\n")

    if args.model:
        print("Loading model (first run downloads ~1.8 GB)...")
        from app.tts.engine import engine
        engine._ensure_loaded()
        print(f"Loaded on {engine.device}")

    failed = [c for c in checks if not c[1]]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="makata",
        description="Makata — zero-shot voice-cloning TTS server (Tagalog/Taglish/English)",
    )
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="start the web server + UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8300)
    serve.add_argument("--engine", choices=("omni", "xtts"), help="engine for new voices (default omni)")
    serve.add_argument("--device", choices=("cuda", "cpu"), help="force compute device (default auto)")
    serve.add_argument("--data-dir", help="override data directory (or env MAKATA_DATA_DIR)")
    serve.add_argument("--reload", action="store_true", help="auto-reload on code changes (dev)")

    check = sub.add_parser("check-setup", help="verify python/torch/cuda/ffmpeg/deps")
    check.add_argument("--model", action="store_true", help="also download/load XTTS v2")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "serve":
        return cmd_serve(args)
    if args.command == "check-setup":
        return cmd_check_setup(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
