"""Pronunciation drill sentences for sharpening a cloned Tagalog voice.

OmniVoice clones best when the prompt contains the phonemes that actually
drift (see README "Recording guidelines"). Each entry below packs one such
sound into a short, sentence-complete clip so the user can record it as an
extra sample. Sentences are intentionally natural Filipino, not tongue-twisters.
"""
from __future__ import annotations

_DRILL = [
    {
        "tag": "mga",
        "text": "Nakita ko ang mga bata at ang mga aso sa parke.",
        "why": "mga is pronounced 'ma-nga', not 'emas'",
        "en": "I saw the children and the dogs in the park.",
    },
    {
        "tag": "ng / ngayon",
        "text": "Ngayon ay kumakain ang buong pamilya ng tanghalian.",
        "why": "word-initial ng and the ng linker (read as 'nang')",
        "en": "Now the whole family is eating lunch.",
    },
    {
        "tag": "rolled r",
        "text": "Maraming mamimili ang nagbabayad ng malaking halaga sa merkado.",
        "why": "tapped/rolled r in marami, mamimili, merkado",
        "en": "Many shoppers pay a large amount at the market.",
    },
    {
        "tag": "glottal stop bata'",
        "text": "Ang bata ay nakasuot ng kanyang bata sa loob ng bahay.",
        "why": "bata (child) vs bata' (robe) — the final glottal stop",
        "en": "The child wore his robe inside the house.",
    },
    {
        "tag": "ñ / Spanish loanwords",
        "text": "Nagpaalam na si Señor at Señorita sa mga bisita ni Niño.",
        "why": "the ñ sound in Señor, Señorita, Niño",
        "en": "Señor and Señorita already said goodbye to Niño's guests.",
    },
    {
        "tag": "loanwords (Filipino forms)",
        "text": "Pupunta kami sa mall, manonood ng YouTube, at gagamit ng kompyuter pagkatapos.",
        "why": "mall/mol, YouTube/yutub, computer/kompyuter",
        "en": "We'll go to the mall, watch YouTube, and use the computer afterwards.",
    },
    {
        "tag": "nang",
        "text": "Umalis ang bata nang mabilis nang hindi napansin ng nanay.",
        "why": "nang as adverb linker and in nang hindi",
        "en": "The child left quickly without the mother noticing.",
    },
    {
        "tag": "question prosody",
        "text": "Saan ka pupunta mamayang gabi, at anong oras ka babalik?",
        "why": "question intonation on saan and anong",
        "en": "Where are you going tonight, and what time will you be back?",
    },
]


def drill_pack(count: int = 0) -> list[dict]:
    if count <= 0:
        count = len(_DRILL)
    return [dict(d) for d in _DRILL[:min(count, len(_DRILL))]]


def drill_max() -> int:
    return len(_DRILL)