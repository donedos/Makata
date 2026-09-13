"""Tagalog pronunciation lexicon (text-rewrite layer before synthesis).

OmniVoice's Filipino checkpoint (~8h of training data) reads standard
orthography well but drifts on English-spelled loanwords, brand names, and
words whose spelled form differs from how Filipinos actually say them. This
module rewrites those surface forms into spoken Filipino so the TTS never has
to guess.

Guarded by MAKATA_PRON_LEX (config.PRON_LEX_ENABLED, default on) and only
applied when the language resolves to 'tl'.

MOST ENTRIES ARE SEED PLACEHOLDERS. Populate the map from the failed-word
audit: any time synthesis mangles a word, add the exact input word as the key
and the correct spoken Filipino form as the value. Keep keys lowercase ASCII;
lookup is case-insensitive and word-boundary safe.
"""
from __future__ import annotations

import re

from app import config

# Exact-word rewrites: input word -> spoken Filipino form.
# Only words that OmniVoice/XTTS mispronounce belong here; if the model already
# reads a word correctly, leave it alone (over-rewriting shifts the timbre).
_TL_PRONOUNCE_MAP: dict[str, str] = {
    "facebook": "peysbuk",
    "youtube": "yutub",
    "mall": "mol",
    "computer": "kompyuter",
    "wifi": "waypai",
    "jeepney": "dyip",
}

_TL_PRONOUNCE_MAP_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _TL_PRONOUNCE_MAP) + r")\b",
    re.IGNORECASE,
)

# Ordered multi-word / hard-case rewrites: (regex, replacement).
# Add phrase-level fixes here (e.g. "barangay hall" -> "bulwagang barangay").
# Replaced before the word map above.
_TL_PRONOUNCE_PATTERNS: list[tuple[str, str]] = []


def apply_pronunciation_lexicon(text: str) -> str:
    if not config.PRON_LEX_ENABLED:
        return text
    for pattern, repl in _TL_PRONOUNCE_PATTERNS:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return _TL_PRONOUNCE_MAP_RE.sub(
        lambda m: _TL_PRONOUNCE_MAP[m.group(1).lower()], text
    )