import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _default_data_dir() -> Path:
    """Source checkout keeps data/ beside the code; installs use a per-user dir."""
    if (PROJECT_ROOT / "pyproject.toml").exists() or (PROJECT_ROOT / ".git").exists():
        return PROJECT_ROOT / "data"
    if sys.platform == "win32":
        base = Path(os.getenv("LOCALAPPDATA", "~/AppData/Local"))
    elif sys.platform == "darwin":
        base = Path("~/Library/Application Support")
    else:
        base = Path(os.getenv("XDG_DATA_HOME", "~/.local/share"))
    return base.expanduser() / "makata"


DATA_DIR = Path(os.getenv("MAKATA_DATA_DIR", "")).expanduser() if os.getenv("MAKATA_DATA_DIR") else _default_data_dir()
REFERENCES_DIR = DATA_DIR / "references"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
OUTPUT_DIR = DATA_DIR / "output"
TMP_DIR = DATA_DIR / "tmp"

SAMPLE_RATE = 24000
TARGET_LUFS = -23.0
MIN_REF_SECONDS = 5.0
MAX_REF_SECONDS = 90.0
WARN_REF_SECONDS = 45.0

XTTS_MODEL = os.getenv("MAKATA_XTTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")

OMNI_MODEL = os.getenv("MAKATA_OMNI_MODEL", "k2-fsa/OmniVoice")
OMNI_TARGET_REF_SECONDS = float(os.getenv("MAKATA_OMNI_TARGET_REF", "12"))
OMNI_NUM_STEP = int(os.getenv("MAKATA_OMNI_NUM_STEP", "48"))
# Multi-sample rebuild: total stitched clone-prompt length cap (README advises 3-20s refs)
OMNI_MAX_PROMPT_SECONDS = float(os.getenv("MAKATA_OMNI_MAX_PROMPT", "20"))
# Silence inserted between stitched samples so joins land on natural pauses
SAMPLE_GAP_SECONDS = 0.25

ENGINES = ("omni", "xtts")
DEFAULT_ENGINE = os.getenv("MAKATA_ENGINE", "omni").strip().lower()
if DEFAULT_ENGINE not in ENGINES:
    DEFAULT_ENGINE = "omni"

XTTS_LANGUAGES = [
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
    "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi",
]

# XTTS has no Tagalog voice; tl text routes through a fallback phonemizer.
# 'es' beats 'en' here: Filipino orthography is Spanish-derived (ñ, rolled r,
# vowel set, ng) so the Spanish phonemizer reads it far closer to native.
TAGALOG_FALLBACK_LANG = os.getenv("MAKATA_TL_FALLBACK", "es")
DEVICE = os.getenv("MAKATA_DEVICE", "auto")

# Rewrite known-mispronounced words into spoken Filipino before synthesis.
PRON_LEX_ENABLED = os.getenv("MAKATA_PRON_LEX", "1").strip().lower() in (
    "1", "true", "yes", "on",
)

TEMPERATURE = 0.65
TOP_P = 0.85
TOP_K = 50
REPETITION_PENALTY = 1.5
LENGTH_PENALTY = 1.0


def ensure_dirs() -> None:
    for d in (DATA_DIR, REFERENCES_DIR, EMBEDDINGS_DIR, OUTPUT_DIR, TMP_DIR):
        d.mkdir(parents=True, exist_ok=True)
