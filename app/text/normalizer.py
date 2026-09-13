"""Text normalization for Tagalog and English before TTS synthesis.

Expands numbers, currency (₱/$), percentages, and ISO dates into spoken
words so the TTS model never has to read raw digits. Tagalog number words
are generated natively (num2words has no `tl` backend).

Note: XTTS v2 has no official `tl` voice; Tagalog text is synthesized via a
fallback language phonemizer (config.TAGALOG_FALLBACK_LANG), so normalized
output stays in standard Filipino orthography.
"""
from __future__ import annotations

import re
import unicodedata

from app.text.pronounce import apply_pronunciation_lexicon

_ONES_EN = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
            "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
            "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS_EN = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES_EN = [(10**9, "billion"), (10**6, "million"), (10**3, "thousand"), (100, "hundred")]

_ONES_TL = ["sero", "isa", "dalawa", "tatlo", "apat", "lima", "anim", "pito", "walo", "siyam",
            "sampu", "labing-isa", "labindalawa", "labintatlo", "labing-apat", "labinlima",
            "labing-anim", "labimpito", "labingwalo", "labinsiyam"]
_TENS_TL = ["", "", "dalawampu", "tatlumpu", "apatnapu", "limampu",
            "anumnapu", "pitumpu", "walumpu", "siyamnapu"]
_SCALES_TL = [(10**9, "bilyon"), (10**6, "milyon"), (10**3, "libo"), (100, "daan")]


def _under_100_en(n: int) -> str:
    if n < 20:
        return _ONES_EN[n]
    t, r = divmod(n, 10)
    return _TENS_EN[t] + ("-" + _ONES_EN[r] if r else "")


def _under_1000_en(n: int) -> str:
    if n < 100:
        return _under_100_en(n)
    h, r = divmod(n, 100)
    return _ONES_EN[h] + " hundred" + (" and " + _under_100_en(r) if r else "")


def _number_to_words_en(n: int) -> str:
    if n == 0:
        return "zero"
    parts: list[str] = []
    for value, name in _SCALES_EN:
        count, n = divmod(n, value)
        if count:
            parts.append(_number_to_words_en(count) + " " + name)
    if n:
        parts.append(_under_100_en(n))
    return " ".join(parts)


def _under_100_tl(n: int) -> str:
    if n < 20:
        return _ONES_TL[n]
    t, r = divmod(n, 10)
    if not r:
        return _TENS_TL[t]
    tens_word = _TENS_TL[t]
    connector = "'t" if tens_word.endswith("pu") else " na "
    ones_word = _ONES_TL[r]
    if connector == "'t":
        return f"{tens_word}'t {ones_word}"
    return f"{tens_word} na {ones_word}"


def _tl_link(word: str) -> str:
    """Linker (na/-ng) for connecting a number word to a following noun."""
    if word and word[-1] in "aeiou":
        return word + "ng"
    return word + " na"


def _tl_link_last(words: str) -> str:
    tokens = words.split()
    tokens[-1] = _tl_link(tokens[-1])
    return " ".join(tokens)


def _under_1000_tl(n: int) -> str:
    if n < 100:
        return _under_100_tl(n)
    h, r = divmod(n, 100)
    linked = _tl_link(_ONES_TL[h])
    head = linked + (" raan" if " na" in linked else " daan")
    if not r:
        return head
    return f"{head} at {_under_100_tl(r)}"


def _number_to_words_tl(n: int) -> str:
    if n == 0:
        return "sero"
    parts: list[str] = []
    for value, name in _SCALES_TL:
        count, n = divmod(n, value)
        if count:
            if value == 1000 and count == 1:
                parts.append("isang libo")
            elif value == 100:
                linked = _tl_link(_number_to_words_tl(count))
                parts.append(linked + (" raan" if " na" in linked else " daan"))
            else:
                parts.append(_tl_link(_number_to_words_tl(count)) + " " + name)
    if n:
        if parts:
            parts.append("at " + _under_1000_tl(n))
        else:
            parts.append(_under_1000_tl(n))
    return " ".join(parts)


def number_to_words(n: int, lang: str) -> str:
    return _number_to_words_tl(n) if lang == "tl" else _number_to_words_en(n)


_MONTHS_TL = ["Enero", "Pebrero", "Marso", "Abril", "Mayo", "Hunyo",
              "Hulyo", "Agosto", "Setyembre", "Oktubre", "Nobyembre", "Disyembre"]
_MONTHS_EN = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]


def date_to_words(year: int, month: int, day: int, lang: str) -> str:
    months = _MONTHS_TL if lang == "tl" else _MONTHS_EN
    day_w = number_to_words(day, lang)
    year_w = number_to_words(year, lang)
    if lang == "tl":
        return f"ika-{day_w} ng {months[month - 1]}, {year_w}"
    return f"{months[month - 1]} {day_w}, {year_w}"


_CURRENCY_TL = {"PHP": "piso", "₱": "piso", "USD": "dolyar", "$": "dolyar",
                "EUR": "euro", "€": "euro"}
_CURRENCY_EN = {"PHP": "pesos", "₱": "pesos", "USD": "dollars", "$": "dollars",
                "EUR": "euros", "€": "euros"}


_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F900-\U0001F9FF\uFE0F]"
)
_MULTISPACE_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def normalize_text(text: str, lang: str = "tl") -> str:
    text = unicodedata.normalize("NFC", text)

    def money(m: re.Match) -> str:
        whole = int(m.group(2))
        cents = m.group(3)
        symbol = m.group(1)
        unit = (_CURRENCY_TL if lang == "tl" else _CURRENCY_EN).get(symbol, "")
        whole_words = number_to_words(whole, lang)
        if lang == "tl" and unit:
            whole_words = _tl_link_last(whole_words)
        out = whole_words + " " + unit
        if cents:
            cents_val = int(cents.ljust(2, "0")[:2])
            cent_unit = "sentimo" if lang == "tl" else "cents"
            cent_words = number_to_words(cents_val, lang)
            if lang == "tl":
                cent_words = _tl_link_last(cent_words) if cents_val >= 10 else _tl_link(cent_words)
            out += " at " + cent_words + " " + cent_unit
        return out

    text = re.sub(r"(₱|\$|€)(\d+)(?:\.(\d{1,2}))?", money, text)

    def percent(m: re.Match) -> str:
        word = "porsiyento" if lang == "tl" else "percent"
        words = number_to_words(int(m.group(1)), lang)
        if lang == "tl":
            words = _tl_link_last(words)
        return words + " " + word

    text = re.sub(r"(\d+)\s?%", percent, text)

    def iso_date(m: re.Match) -> str:
        try:
            return date_to_words(int(m.group(1)), int(m.group(2)), int(m.group(3)), lang)
        except (IndexError, ValueError):
            return m.group(0)

    text = re.sub(r"(\d{4})-(\d{2})-(\d{2})\b", iso_date, text)

    def plain_number(m: re.Match) -> str:
        digits = m.group(0).replace(",", "").replace("_", "")
        if len(digits.replace(".", "")) > 12:
            return m.group(0)
        if "." in digits:
            int_part, frac = digits.split(".", 1)
            point = "tuldok" if lang == "tl" else "point"
            frac_words = " ".join(number_to_words(int(d), lang) for d in frac if d.isdigit())
            return number_to_words(int(int_part or 0), lang) + " " + point + " " + frac_words
        return number_to_words(int(digits), lang)

    text = re.sub(r"\d[\d,]*(?:\.\d+)?", plain_number, text)

    for sym, spoken in {"&": "at" if lang == "tl" else "and",
                        "@": "sa" if lang == "tl" else "at",
                        "₱": "", "%": ""}.items():
        text = text.replace(sym, spoken)

    if lang == "tl":
        text = apply_pronunciation_lexicon(text)

    text = _EMOJI_RE.sub("", text)
    text = "".join(c for c in text if c.isprintable() or c == "\n")
    text = _MULTISPACE_RE.sub(" ", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


_TAGALOG_MARKERS = {
    "ang", "ng", "mga", "si", "ni", "kay", "ako", "ikaw", "siya", "kami", "tayo", "kayo",
    "sila", "hindi", "oo", "po", "opo", "ba", "na", "pa", "din", "rin", "lang", "dahil",
    "kung", "pero", "at", "sa", "nasa", "ito", "iyan", "iyon", "ano", "alin", "sino",
    "maganda", "salamat", "walang", "may", "wala", "gusto", "kailangan", "naman", "nga",
}
_ENGLISH_MARKERS = {
    "the", "is", "are", "was", "were", "and", "of", "to", "in", "that", "it", "for",
    "with", "this", "have", "from", "you", "we", "they", "will", "would", "can",
}


def detect_language(text: str) -> str:
    """Return 'tl' or 'en' based on stopword-marker scoring (ties -> 'tl')."""
    tokens = re.findall(r"[a-zA-ZÀ-ÿ']+", unicodedata.normalize("NFC", text.lower()))
    tl_score = sum(1 for t in tokens if t in _TAGALOG_MARKERS)
    en_score = sum(1 for t in tokens if t in _ENGLISH_MARKERS)
    return "en" if en_score > tl_score else "tl"
