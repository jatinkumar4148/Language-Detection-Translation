"""
LinguaDetect AI — Language Detection & Translation
====================================================
Production-hardened version of the original prototype notebook.

Preserves the existing ML approach:
    TextCleaner -> Character TF-IDF -> MultinomialNB   (22 trained languages)
    + FastText lid.176 as a second, independent detector
    + Helsinki-NLP / MarianMT for translation (3-level fallback)

What changed vs. the original notebook and why is documented in README.md
under "Modification Log". The short version:
    - Fixed a source-language-code mapping bug (see NB_LANGUAGE_CODES below)
    - Replaced blind threshold logic with a transparent confidence-fusion score
    - Added script-aware analysis and mixed-language warnings
    - Added caching, error handling, HTML escaping, and a working copy button
    - Did NOT retrain the model and did NOT change NB_language_pipeline.pkl
"""

from __future__ import annotations

import html
import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

import joblib
import numpy as np
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin

# ─────────────────────────────────────────────────────────────────────────
# PAGE CONFIG (must be the first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LinguaDetect AI",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────
# TEXT CLEANER  — must stay byte-for-byte identical to the class used when
# NB_language_pipeline.pkl was trained. The pickle references this class as
# `__main__.TextCleaner`, and Streamlit runs this file as `__main__`, so
# defining it here (not importing it from a helper module) is required for
# joblib.load() to succeed.
# ─────────────────────────────────────────────────────────────────────────
class TextCleaner(BaseEstimator, TransformerMixin):
    """Original preprocessing step baked into the trained NB pipeline.

    Left untouched on purpose: changing this would silently invalidate the
    existing .pkl (the vectorizer downstream was fit on this exact cleaning
    behaviour). Unicode letters (Hindi, Tamil, Arabic, CJK, etc.) are never
    stripped — only ASCII punctuation, digits, URLs and emails are removed.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [self.clean(text) for text in X]

    def clean(self, text: str) -> str:
        text = str(text)
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"\S+@\S+", "", text)
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


# ─────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────

# The exact 22 language labels the NB pipeline was trained on, mapped to
# ISO codes. THIS IS A FIX: the original notebook's `language_codes` dict
# (40 entries) did not match the model's actual 22 classes_ — it was
# missing Estonian, Latin and Pushto, and contained 21 languages the model
# was never trained on (German, Italian, Kannada, Bengali, ...). Any of
# those NB predictions previously fell through to `.get(detected)` -> None.
# This dict is verified against `nb_pipeline.classes_` directly.
NB_LANGUAGE_CODES: dict[str, str] = {
    "Arabic": "ar", "Chinese": "zh", "Dutch": "nl", "English": "en",
    "Estonian": "et", "French": "fr", "Hindi": "hi", "Indonesian": "id",
    "Japanese": "ja", "Korean": "ko", "Latin": "la", "Persian": "fa",
    "Portugese": "pt", "Pushto": "ps", "Romanian": "ro", "Russian": "ru",
    "Spanish": "es", "Swedish": "sv", "Tamil": "ta", "Thai": "th",
    "Turkish": "tr", "Urdu": "ur",
}
NB_CODE_SET = set(NB_LANGUAGE_CODES.values())
N_NB_CLASSES = len(NB_LANGUAGE_CODES)
# If NB's top-1 probability sits at/near 1/22 (the class prior), the
# TF-IDF vector for this input had ~no overlap with the training
# vocabulary (e.g. an untrained script). Verified empirically: digits,
# emoji, and out-of-vocabulary scripts all produce a near-uniform
# posterior around 0.045-0.047. 1.8x the prior is a safe cut above that
# noise floor while still catching genuinely weak-but-real signal.
NB_UNINFORMATIVE_THRESHOLD = (1 / N_NB_CLASSES) * 1.8

# Human-readable display names for ISO codes, used for BOTH the NB model's
# 22 languages and whatever FastText's 176 languages return, so the UI
# never shows a raw ISO code as the "detected language" name. Also fixes
# the "Portugese" -> "Portuguese" spelling issue while preserving the
# internal 'pt' code used everywhere else.
DISPLAY_NAMES: dict[str, str] = {
    "af": "Afrikaans", "am": "Amharic", "ar": "Arabic", "as": "Assamese",
    "az": "Azerbaijani", "be": "Belarusian", "bg": "Bulgarian", "bn": "Bengali",
    "bo": "Tibetan", "bs": "Bosnian", "ca": "Catalan", "ceb": "Cebuano",
    "co": "Corsican", "cs": "Czech", "cy": "Welsh", "da": "Danish",
    "de": "German", "el": "Greek", "en": "English", "eo": "Esperanto",
    "es": "Spanish", "et": "Estonian", "eu": "Basque", "fa": "Persian",
    "fi": "Finnish", "fr": "French", "ga": "Irish", "gd": "Scottish Gaelic",
    "gl": "Galician", "gu": "Gujarati", "he": "Hebrew", "hi": "Hindi",
    "hr": "Croatian", "ht": "Haitian Creole", "hu": "Hungarian",
    "hy": "Armenian", "id": "Indonesian", "is": "Icelandic", "it": "Italian",
    "ja": "Japanese", "jv": "Javanese", "ka": "Georgian", "kk": "Kazakh",
    "km": "Khmer", "kn": "Kannada", "ko": "Korean", "ku": "Kurdish",
    "ky": "Kyrgyz", "la": "Latin", "lb": "Luxembourgish", "lo": "Lao",
    "lt": "Lithuanian", "lv": "Latvian", "mg": "Malagasy", "mk": "Macedonian",
    "ml": "Malayalam", "mn": "Mongolian", "mr": "Marathi", "ms": "Malay",
    "mt": "Maltese", "my": "Burmese", "ne": "Nepali", "nl": "Dutch",
    "no": "Norwegian", "or": "Odia", "pa": "Punjabi", "pl": "Polish",
    "ps": "Pashto", "pt": "Portuguese", "ro": "Romanian", "ru": "Russian",
    "sa": "Sanskrit", "sd": "Sindhi", "si": "Sinhala", "sk": "Slovak",
    "sl": "Slovenian", "so": "Somali", "sq": "Albanian", "sr": "Serbian",
    "su": "Sundanese", "sv": "Swedish", "sw": "Swahili", "ta": "Tamil",
    "te": "Telugu", "tg": "Tajik", "th": "Thai", "tk": "Turkmen",
    "tl": "Tagalog", "tr": "Turkish", "tt": "Tatar", "ug": "Uyghur",
    "uk": "Ukrainian", "ur": "Urdu", "uz": "Uzbek", "vi": "Vietnamese",
    "wa": "Walloon", "xh": "Xhosa", "yi": "Yiddish", "yo": "Yoruba",
    "yue": "Cantonese", "zh": "Chinese", "zu": "Zulu",
}
FLAGS: dict[str, str] = {
    "en": "🇬🇧", "fr": "🇫🇷", "es": "🇪🇸", "de": "🇩🇪", "it": "🇮🇹", "pt": "🇵🇹",
    "ru": "🇷🇺", "nl": "🇳🇱", "sv": "🇸🇪", "tr": "🇹🇷", "ar": "🇸🇦", "hi": "🇮🇳",
    "ta": "🇮🇳", "ur": "🇵🇰", "fa": "🇮🇷", "zh": "🇨🇳", "ja": "🇯🇵", "ko": "🇰🇷",
    "id": "🇮🇩", "th": "🇹🇭", "vi": "🇻🇳", "bn": "🇧🇩", "pl": "🇵🇱", "uk": "🇺🇦",
    "el": "🇬🇷", "he": "🇮🇱", "ro": "🇷🇴", "cs": "🇨🇿", "hu": "🇭🇺", "fi": "🇫🇮",
    "da": "🇩🇰", "no": "🇳🇴", "et": "🇪🇪", "ps": "🇦🇫", "la": "🏛️",
}


def display_name(code: Optional[str]) -> str:
    if not code:
        return "Unknown"
    return DISPLAY_NAMES.get(code, code.upper())


def flag_for(code: Optional[str]) -> str:
    return FLAGS.get(code, "🌐")


# Target-language options for translation. Broader than the NB's 22 trained
# source languages (translation TARGET doesn't need to be something the NB
# model can detect — it only needs Helsinki-NLP/MarianMT support, which is
# checked at call time with graceful fallback, never assumed in advance).
TRANSLATE_TARGET_CODES: dict[str, str] = {
    "Arabic": "ar", "Bengali": "bn", "Bulgarian": "bg", "Catalan": "ca",
    "Chinese": "zh", "Croatian": "hr", "Czech": "cs", "Danish": "da",
    "Dutch": "nl", "English": "en", "Estonian": "et", "Finnish": "fi",
    "French": "fr", "German": "de", "Greek": "el", "Hebrew": "he",
    "Hindi": "hi", "Hungarian": "hu", "Indonesian": "id", "Italian": "it",
    "Japanese": "ja", "Kannada": "kn", "Korean": "ko", "Latin": "la",
    "Macedonian": "mk", "Malayalam": "ml", "Persian": "fa", "Polish": "pl",
    "Portuguese": "pt", "Pashto": "ps", "Romanian": "ro", "Russian": "ru",
    "Serbian": "sr", "Spanish": "es", "Swahili": "sw", "Swedish": "sv",
    "Tamil": "ta", "Telugu": "te", "Thai": "th", "Turkish": "tr",
    "Ukrainian": "uk", "Urdu": "ur", "Vietnamese": "vi",
}

# Unicode script ranges used for the lightweight script-analysis layer.
# Deliberately coarse: a script narrows down a *family* of plausible
# languages, it never picks a single language on its own (several
# languages share the same script).
SCRIPT_RANGES: list[tuple[str, list[tuple[int, int]], list[str]]] = [
    ("Devanagari", [(0x0900, 0x097F)], ["hi", "mr", "ne", "sa"]),
    ("Bengali", [(0x0980, 0x09FF)], ["bn", "as"]),
    ("Gurmukhi", [(0x0A00, 0x0A7F)], ["pa"]),
    ("Gujarati", [(0x0A80, 0x0AFF)], ["gu"]),
    ("Tamil", [(0x0B80, 0x0BFF)], ["ta"]),
    ("Telugu", [(0x0C00, 0x0C7F)], ["te"]),
    ("Kannada", [(0x0C80, 0x0CFF)], ["kn"]),
    ("Malayalam", [(0x0D00, 0x0D7F)], ["ml"]),
    ("Sinhala", [(0x0D80, 0x0DFF)], ["si"]),
    ("Thai", [(0x0E00, 0x0E7F)], ["th"]),
    ("Hebrew", [(0x0590, 0x05FF)], ["he"]),
    ("Arabic", [(0x0600, 0x06FF), (0x0750, 0x077F)], ["ar", "ur", "fa", "ps"]),
    ("Cyrillic", [(0x0400, 0x04FF)], ["ru", "uk", "bg", "sr", "mk", "be"]),
    ("Greek", [(0x0370, 0x03FF)], ["el"]),
    ("Hangul", [(0xAC00, 0xD7A3)], ["ko"]),
    ("Hiragana", [(0x3040, 0x309F)], ["ja"]),
    ("Katakana", [(0x30A0, 0x30FF)], ["ja"]),
    ("CJK", [(0x4E00, 0x9FFF)], ["zh", "ja"]),
    ("Latin", [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F)],
     ["en", "fr", "es", "de", "it", "pt", "nl", "sv", "tr", "id", "vi", "la",
      "ro", "pl", "cs", "hu", "da", "no", "fi", "et", "hr"]),
]


# ─────────────────────────────────────────────────────────────────────────
# TEXT ANALYSIS HELPERS
# ─────────────────────────────────────────────────────────────────────────
def extract_linguistic_content(text: str) -> str:
    """Strip URLs, emails, digits, emoji and punctuation, keep only letters
    (any script) and marks. Used to decide whether there is anything to
    detect at all — a URL-only or emoji-only input has none."""
    t = re.sub(r"http\S+|www\S+", "", text)
    t = re.sub(r"\S+@\S+", "", t)
    t = re.sub(r"\d+", "", t)
    kept = [ch for ch in t if unicodedata.category(ch).startswith(("L", "M")) or ch.isspace()]
    return re.sub(r"\s+", " ", "".join(kept)).strip()


def analyze_script(text: str) -> dict:
    """Count characters per Unicode script family and report the dominant
    one plus whether the input looks like a mix of two or more scripts."""
    counts: dict[str, int] = {name: 0 for name, _, _ in SCRIPT_RANGES}
    total = 0
    for ch in text:
        cp = ord(ch)
        if ch.isspace() or unicodedata.category(ch) not in (
            "Lu", "Ll", "Lt", "Lm", "Lo", "Mn", "Mc",
        ):
            continue
        total += 1
        for name, ranges, _ in SCRIPT_RANGES:
            if any(lo <= cp <= hi for lo, hi in ranges):
                counts[name] += 1
                break

    counts = {k: v for k, v in counts.items() if v > 0}
    if total == 0 or not counts:
        return {"dominant": None, "shares": {}, "mixed": False, "plausible_codes": []}

    shares = {k: v / total for k, v in counts.items()}
    dominant = max(shares, key=shares.get)
    secondary = [k for k, v in shares.items() if k != dominant and v >= 0.15]
    plausible = next((codes for name, _, codes in SCRIPT_RANGES if name == dominant), [])

    return {
        "dominant": dominant,
        "shares": shares,
        "mixed": len(secondary) > 0,
        "secondary": secondary,
        "plausible_codes": plausible,
    }


# ─────────────────────────────────────────────────────────────────────────
# MODEL LOADING (cached — loaded once per process, not per rerun)
# ─────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_nb_model():
    return joblib.load("NB_language_pipeline.pkl")


@st.cache_resource(show_spinner=False)
def load_fasttext_model():
    import fasttext

    model_path = "lid.176.ftz"
    if not os.path.exists(model_path):
        import urllib.request

        urllib.request.urlretrieve(
            "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz",
            model_path,
        )
    fasttext.FastText.eprint = lambda *_: None
    return fasttext.load_model(model_path)


@st.cache_resource(show_spinner=False)
def load_translation_model(source_code: str, target_code: str):
    """Cached per (source_code, target_code) pair. Tries, in order:
    direct opus-mt model -> tc-big opus-mt model. Returns
    (tokenizer, model, model_name) or raises if neither exists.
    Loading happens lazily — only when a translation is actually requested,
    never at app startup.
    """
    from transformers import MarianMTModel, MarianTokenizer

    candidates = [
        f"Helsinki-NLP/opus-mt-{source_code}-{target_code}",
        f"Helsinki-NLP/opus-mt-tc-big-{source_code}-{target_code}",
    ]
    last_error = None
    for model_name in candidates:
        try:
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            return tokenizer, model, model_name
        except Exception as exc:  # noqa: BLE001 — deliberately broad, we try the next candidate
            last_error = exc
            continue
    raise RuntimeError(f"No direct model for {source_code}->{target_code}") from last_error


# ─────────────────────────────────────────────────────────────────────────
# DETECTION PIPELINE
# ─────────────────────────────────────────────────────────────────────────
@dataclass
class NBResult:
    label: Optional[str]
    code: Optional[str]
    top1: float
    margin: float
    informative: bool
    top5: list  # list[(label, prob)]


@dataclass
class FTResult:
    code: Optional[str]
    confidence: float
    ok: bool
    top3: list = field(default_factory=list)


@dataclass
class DetectionResult:
    code: Optional[str]
    name: str
    confidence_score: float          # 0-1 composite, NOT a calibrated probability
    confidence_pct: float            # for display
    status: str                      # High / Medium / Low / Unable to Detect
    method: str                      # label shown in the UI
    nb: NBResult
    ft: FTResult
    agree: bool
    notes: list
    script: dict
    mixed_warning: bool


def detect_with_nb(text: str, nb_pipeline) -> NBResult:
    probs = nb_pipeline.predict_proba([text])[0]
    pairs = sorted(zip(nb_pipeline.classes_, probs), key=lambda x: -x[1])
    top1_label, top1 = pairs[0]
    margin = float(top1 - pairs[1][1])
    informative = top1 >= NB_UNINFORMATIVE_THRESHOLD
    return NBResult(
        label=top1_label,
        code=NB_LANGUAGE_CODES.get(top1_label),
        top1=float(top1),
        margin=margin,
        informative=informative,
        top5=[(lbl, float(p)) for lbl, p in pairs[:5]],
    )


def detect_with_fasttext(text: str, ft_model) -> FTResult:
    try:
        clean = text.replace("\n", " ").strip()
        labels, scores = ft_model.predict(clean, k=3)
        top3 = [(lbl.replace("__label__", ""), float(sc)) for lbl, sc in zip(labels, scores)]
        return FTResult(code=top3[0][0], confidence=top3[0][1], ok=True, top3=top3)
    except Exception:  # noqa: BLE001
        return FTResult(code=None, confidence=0.0, ok=False, top3=[])


def calculate_confidence(text: str, nb: NBResult, ft: FTResult) -> tuple[float, str, str, list]:
    """Combine NB + FastText + agreement + margin + length into one 0-1
    composite score and pick a method label. This is a heuristic fusion,
    not a calibrated probability — documented as such in the UI.

    Empirically-motivated design decisions (see README, "Detection Design
    Notes", for the measurements behind these):
      * If FastText's language isn't one of NB's 22 trained classes, NB is
        structurally unable to be correct — trust FastText outright rather
        than average two numbers where one is guaranteed noise.
      * Agreement between two independently-trained models is itself
        strong evidence, especially for short inputs where each model's
        raw probability is individually weak (e.g. "Hello").
      * A near-uniform NB posterior (see NB_UNINFORMATIVE_THRESHOLD) means
        "no signal", not "low confidence in a real answer" — it's
        excluded from the score rather than dragging it down.
    """
    ft_ok = ft.ok and ft.confidence > 0.0
    ft_outside_nb = ft_ok and ft.code not in NB_CODE_SET
    agree = nb.informative and ft_ok and (nb.code == ft.code)
    notes: list[str] = []
    meaningful_len = len(extract_linguistic_content(text))

    if ft_outside_nb:
        score = 0.35 + min(ft.confidence, 1.0) * 0.55
        notes.append(
            "FastText's prediction is outside the custom NB model's 22 trained "
            "languages, so the NB output is not usable here — FastText result used."
        )
        final_code = ft.code
        method = "FastText Fallback (outside NB training set)"
    else:
        score = 0.40 * (nb.top1 if nb.informative else 0.0)
        score += 0.35 * (ft.confidence if ft_ok else 0.0)
        score += 0.10 * (min(nb.margin / 0.3, 1.0) if nb.informative else 0.0)

        if agree:
            score += 0.20
            notes.append("NB and FastText independently agree.")
            if meaningful_len < 10:
                score += 0.08
                notes.append("Short input, but agreement across both models offsets that.")
        elif nb.informative and ft_ok:
            score -= 0.12
            notes.append("NB and FastText disagree.")

        if meaningful_len < 5:
            score -= 0.15
            notes.append("Very short input reduces reliability.")
        elif meaningful_len >= 20:
            score += 0.05

        if not nb.informative:
            notes.append("NB found no informative signal for this text (unseen script/vocabulary).")

        if agree:
            final_code = nb.code
            method = "NB + FastText Verified"
        elif nb.informative and nb.top1 >= 0.60 and (not ft_ok or nb.top1 >= ft.confidence):
            final_code = nb.code
            method = "NB High Confidence"
        elif ft_ok and ft.confidence >= 0.45:
            final_code = ft.code
            method = "FastText Fallback"
        elif nb.informative:
            final_code = nb.code
            method = "NB Low Confidence"
        else:
            final_code, method = None, "Unable to Detect"

    score = max(0.0, min(1.0, score))
    if meaningful_len < 2:
        return 0.0, None, "No linguistic content", ["Input has no detectable letters."]
    if score >= 0.72:
        status = "High Confidence"
    elif score >= 0.42:
        status = "Medium Confidence"
    elif score >= 0.20:
        status = "Low Confidence"
    else:
        status = "Unable to Detect"
        final_code = None

    return score, final_code, (method if final_code else "Unable to Detect"), notes


def combine_detection_results(text: str, nb_pipeline, ft_model) -> DetectionResult:
    linguistic = extract_linguistic_content(text)
    script_info = analyze_script(text)

    if len(linguistic) < 2:
        empty_nb = NBResult(None, None, 0.0, 0.0, False, [])
        empty_ft = FTResult(None, 0.0, False, [])
        return DetectionResult(
            code=None, name="Unable to Detect", confidence_score=0.0, confidence_pct=0.0,
            status="Unable to Detect", method="No linguistic content",
            nb=empty_nb, ft=empty_ft, agree=False,
            notes=["No detectable letters found — only numbers, symbols, URLs, or emoji."],
            script=script_info, mixed_warning=False,
        )

    nb = detect_with_nb(text, nb_pipeline)
    ft = detect_with_fasttext(text, ft_model)
    score, code, method, notes = calculate_confidence(text, nb, ft)
    status = (
        "High Confidence" if score >= 0.72 else
        "Medium Confidence" if score >= 0.42 else
        "Low Confidence" if score >= 0.20 else
        "Unable to Detect"
    )
    agree = nb.informative and ft.ok and (nb.code == ft.code)

    mixed_warning = bool(script_info.get("mixed")) and len(linguistic) >= 6

    return DetectionResult(
        code=code,
        name=display_name(code) if code else "Unable to Detect",
        confidence_score=score,
        confidence_pct=round(score * 100, 1),
        status=status,
        method=method,
        nb=nb,
        ft=ft,
        agree=agree,
        notes=notes,
        script=script_info,
        mixed_warning=mixed_warning,
    )


# ─────────────────────────────────────────────────────────────────────────
# TRANSLATION
# ─────────────────────────────────────────────────────────────────────────
def translate_text(text: str, source_code: str, target_code: str) -> tuple[str, str]:
    """Returns (translated_text_or_message, status) where status is one of
    'ok', 'same', 'unavailable', 'error'. Never raises — all Marian/HF
    exceptions are caught so the app keeps running."""
    if not source_code or not target_code:
        return "Translation unavailable: language code missing.", "unavailable"
    if source_code == target_code:
        return text, "same"

    # Level 1 + 2: direct / tc-big (handled inside load_translation_model)
    try:
        tokenizer, model, _ = load_translation_model(source_code, target_code)
        tokens = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512)
        out = model.generate(**tokens, max_length=512)
        return tokenizer.decode(out[0], skip_special_tokens=True), "ok"
    except Exception:  # noqa: BLE001
        pass

    # Level 3: pivot through English
    try:
        if source_code != "en":
            tok1, m1, _ = load_translation_model(source_code, "en")
            t1 = tok1([text], return_tensors="pt", padding=True, truncation=True, max_length=512)
            english_text = tok1.decode(m1.generate(**t1, max_length=512)[0], skip_special_tokens=True)
        else:
            english_text = text

        if target_code == "en":
            return english_text, "ok"

        tok2, m2, _ = load_translation_model("en", target_code)
        t2 = tok2([english_text], return_tensors="pt", padding=True, truncation=True, max_length=512)
        return tok2.decode(m2.generate(**t2, max_length=512)[0], skip_special_tokens=True), "ok"
    except Exception:  # noqa: BLE001
        return (
            "A translation model for this language pair is currently unavailable. "
            "Try English or another widely-supported target language.",
            "unavailable",
        )


# ─────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg: #0a0a0f; --surface: #12121a; --card: #1a1a26; --border: #2a2a3d;
    --accent: #6c63ff; --accent2: #ff6584; --accent3: #43e97b;
    --warn: #ffb74d; --text: #e8e8f0; --muted: #6b6b8a;
}
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background: var(--bg) !important; color: var(--text) !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem !important; max-width: 1200px; }

.hero { text-align: center; padding: 3rem 0 2rem; }
.hero-badge { display: inline-block; background: linear-gradient(135deg, #6c63ff22, #ff658422); border: 1px solid #6c63ff44; border-radius: 100px; padding: 6px 18px; font-size: 0.75rem; letter-spacing: 2px; text-transform: uppercase; color: var(--accent); margin-bottom: 1.5rem; }
.hero-title { font-family: 'Syne', sans-serif; font-size: clamp(2.5rem, 5vw, 4rem); font-weight: 800; background: linear-gradient(135deg, #fff 0%, #6c63ff 50%, #ff6584 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1.1; margin-bottom: 1rem; }
.hero-sub { color: var(--muted); font-size: 1.1rem; font-weight: 300; max-width: 560px; margin: 0 auto 2rem; }

.stats-bar { display: flex; justify-content: center; gap: 3rem; margin: 1.5rem 0 2.5rem; flex-wrap: wrap; }
.stat-item { text-align: center; }
.stat-number { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 700; background: linear-gradient(135deg, #6c63ff, #ff6584); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.stat-label { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; }

.glass-card { background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem; }
.section-label { font-size: 0.7rem; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); margin-bottom: 0.5rem; }

.stTextArea textarea { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; color: var(--text) !important; font-family: 'DM Sans', sans-serif !important; font-size: 1rem !important; padding: 1rem !important; }
.stTextArea textarea:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 3px #6c63ff22 !important; }
.stSelectbox > div > div { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; color: var(--text) !important; }

.stButton > button { background: linear-gradient(135deg, #6c63ff, #9c88ff) !important; color: white !important; border: none !important; border-radius: 12px !important; padding: 0.7rem 1.5rem !important; font-family: 'Syne', sans-serif !important; font-weight: 600 !important; font-size: 0.95rem !important; width: 100% !important; transition: all 0.25s !important; }
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 25px #6c63ff44 !important; }

.counter-row { display:flex; gap:1.2rem; font-size:0.78rem; color:var(--muted); margin: 0.4rem 0 0.2rem; }

.result-detected { background: linear-gradient(135deg, #6c63ff11, #9c88ff11); border: 1px solid #6c63ff44; border-radius: 12px; padding: 1.2rem 1.5rem; margin-bottom: 1rem; }
.result-translated { background: linear-gradient(135deg, #43e97b11, #38f9d711); border: 1px solid #43e97b44; border-radius: 12px; padding: 1.2rem 1.5rem; margin-bottom: 1rem; }
.result-label { font-size: 0.7rem; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 0.4rem; }
.result-detected .result-label { color: #9c88ff; }
.result-translated .result-label { color: #43e97b; }
.result-value { font-family: 'Syne', sans-serif; font-size: 1.35rem; font-weight: 700; color: var(--text); word-break: break-word; }
.result-sub { font-size: 0.8rem; color: var(--muted); margin-top: 0.3rem; }

.prob-container { margin-top: 0.5rem; }
.prob-row { display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.55rem; }
.prob-lang { width: 100px; font-size: 0.83rem; color: var(--text); text-align: right; flex-shrink:0; }
.prob-bar-bg { flex: 1; height: 8px; background: var(--surface); border-radius: 100px; overflow: hidden; }
.prob-bar-fill { height: 100%; border-radius: 100px; background: linear-gradient(90deg, #6c63ff, #9c88ff); }
.prob-pct { width: 48px; font-size: 0.78rem; color: var(--muted); text-align: right; flex-shrink:0; }

.method-badge { display: inline-flex; align-items: center; gap: 6px; background: #43e97b22; border: 1px solid #43e97b44; border-radius: 100px; padding: 4px 12px; font-size: 0.75rem; color: #43e97b; margin-bottom: 1rem; }
.method-badge-warn { background: #ffb74d22; border-color: #ffb74d44; color: #ffb74d; }
.mixed-warning { background: #ffb74d15; border: 1px solid #ffb74d44; border-radius: 10px; padding: 0.7rem 1rem; font-size: 0.82rem; color: var(--warn); margin-bottom: 1rem; }
.transparency-card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 0.9rem 1.1rem; font-size: 0.82rem; margin-top: 0.6rem; }
.transparency-row { display:flex; justify-content: space-between; padding: 2px 0; color: var(--muted); }
.transparency-row b { color: var(--text); font-weight: 500; }

.info-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 2rem; }
.info-card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1.2rem; text-align: center; }
.info-icon { font-size: 1.8rem; margin-bottom: 0.5rem; }
.info-title { font-family: 'Syne', sans-serif; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.3rem; }
.info-desc { font-size: 0.78rem; color: var(--muted); line-height: 1.5; }

.divider { border: none; border-top: 1px solid var(--border); margin: 2rem 0; }
.stSpinner > div { border-top-color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────
# RENDER HELPERS
# ─────────────────────────────────────────────────────────────────────────
def render_probability_bars(top5: list) -> str:
    rows = ""
    for label, prob in top5:
        name = DISPLAY_NAMES.get(NB_LANGUAGE_CODES.get(label), label)
        width = round(prob * 100, 1)
        rows += f"""
        <div class="prob-row">
            <div class="prob-lang">{html.escape(name)}</div>
            <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:{width}%"></div></div>
            <div class="prob-pct">{width:.1f}%</div>
        </div>"""
    return f'<div class="prob-container">{rows}</div>'


def render_copy_button(text: str, key: str) -> None:
    """A copy button that actually works, via the browser clipboard API.
    Text is safely JSON-encoded so quotes/newlines/HTML can't break the
    embedded script or be interpreted as markup."""
    import streamlit.components.v1 as components

    safe_text = json.dumps(text)
    components.html(
        f"""
        <div style="font-family:'DM Sans',sans-serif;">
          <button id="copy-{key}" style="
              background:#1a1a26;border:1px solid #2a2a3d;border-radius:8px;
              padding:6px 14px;font-size:0.78rem;color:#9c88ff;cursor:pointer;">
            📋 Copy translation
          </button>
          <script>
            const btn = document.getElementById("copy-{key}");
            btn.addEventListener("click", async () => {{
              try {{
                await navigator.clipboard.writeText({safe_text});
                btn.innerText = "✅ Copied!";
                setTimeout(() => btn.innerText = "📋 Copy translation", 1500);
              }} catch (e) {{
                btn.innerText = "⚠️ Copy failed";
              }}
            }});
          </script>
        </div>
        """,
        height=42,
    )


# ─────────────────────────────────────────────────────────────────────────
# SESSION STATE CALLBACKS
# ─────────────────────────────────────────────────────────────────────────
EXAMPLES = {
    "🇪🇸 Spanish": "Hola, ¿cómo estás? Me llamo Juan y soy de España.",
    "🇫🇷 French": "Bonjour, comment allez-vous? Je suis étudiant.",
    "🇩🇪 German": "Guten Morgen! Wie geht es Ihnen heute?",
    "🇯🇵 Japanese": "こんにちは、お元気ですか？私は学生です。",
    "🇮🇳 Hindi": "नमस्ते! आप कैसे हैं? मैं ठीक हूं।",
}

if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "result" not in st.session_state:
    st.session_state.result = None
if "translation" not in st.session_state:
    st.session_state.translation = None


def set_example(example_text: str) -> None:
    st.session_state.input_text = example_text
    st.session_state.result = None
    st.session_state.translation = None


def clear_input() -> None:
    st.session_state.input_text = ""
    st.session_state.result = None
    st.session_state.translation = None


# ─────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🤖 ML + NLP Project</div>
    <div class="hero-title">LinguaDetect AI</div>
    <div class="hero-sub">Detect any language instantly and translate it using a
    custom-trained classifier, FastText verification, and Helsinki-NLP transformers.</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="stats-bar">
    <div class="stat-item"><div class="stat-number">22</div><div class="stat-label">ML-Trained Languages</div></div>
    <div class="stat-item"><div class="stat-number">176</div><div class="stat-label">FastText-Supported Languages</div></div>
    <div class="stat-item"><div class="stat-number">97.57%</div><div class="stat-label">NB Test Accuracy</div></div>
    <div class="stat-item"><div class="stat-number">3</div><div class="stat-label">Translation Fallback Levels</div></div>
</div>
""", unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────
models_ok = True
try:
    with st.spinner("Loading ML models..."):
        nb_pipeline = load_nb_model()
        ft_model = load_fasttext_model()
except Exception as exc:  # noqa: BLE001
    st.error("❌ Could not load the detection models.")
    st.caption(f"Details: {html.escape(str(exc))}")
    st.info("Make sure NB_language_pipeline.pkl and lid.176.ftz are in the app folder.")
    models_ok = False

# ─────────────────────────────────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1.1, 0.9], gap="large")

with col1:
    #st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">✍️ Input Text</div>', unsafe_allow_html=True)

    st.text_area(
        label="Text to detect",
        height=170,
        placeholder="Type or paste text in any language...\n\nExample: Bonjour, comment allez-vous?\nOr: नमस्ते, आप कैसे हैं?",
        label_visibility="collapsed",
        key="input_text",
    )

    current_text = st.session_state.input_text
    char_count = len(current_text)
    word_count = len(current_text.split()) if current_text.strip() else 0
    st.markdown(
        f'<div class="counter-row"><span>Characters: {char_count}</span>'
        f'<span>Words: {word_count}</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label" style="margin-top:1rem">🌍 Translate To</div>', unsafe_allow_html=True)
    target_options = sorted(TRANSLATE_TARGET_CODES.keys())
    target_lang = st.selectbox(
        label="Target language",
        options=target_options,
        index=target_options.index("English"),
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    b1, b2 = st.columns([3, 1])
    with b1:
        translate_btn = st.button("🚀 Detect & Translate", use_container_width=True, disabled=not models_ok)
    with b2:
        st.button("🗑️ Clear", use_container_width=True, on_click=clear_input)

    #st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:1rem">⚡ Try Examples</div>', unsafe_allow_html=True)
    ex_cols = st.columns(len(EXAMPLES))
    for i, (label, example) in enumerate(EXAMPLES.items()):
        with ex_cols[i]:
            st.button(label, key=f"ex_{i}", use_container_width=True,
                      on_click=set_example, args=(example,))

# ─────────────────────────────────────────────────────────────────────────
# RUN DETECTION / TRANSLATION (only on explicit button click)
# ─────────────────────────────────────────────────────────────────────────
if translate_btn and models_ok:
    text = st.session_state.input_text
    if not text or len(text.strip()) < 1:
        st.session_state.result = "empty"
        st.session_state.translation = None
    else:
        with st.spinner("Analyzing language..."):
            result = combine_detection_results(text, nb_pipeline, ft_model)
        st.session_state.result = result
        st.session_state.translation = None

        if result.code:
            if result.code == TRANSLATE_TARGET_CODES.get(target_lang):
                st.session_state.translation = (text, "same")
            else:
                with st.spinner("Translating..."):
                    translated, tstatus = translate_text(
                        text, result.code, TRANSLATE_TARGET_CODES.get(target_lang)
                    )
                st.session_state.translation = (translated, tstatus)

with col2:
    #st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">📊 Results</div>', unsafe_allow_html=True)

    result = st.session_state.result

    if result is None:
        st.markdown("""
        <div style="text-align:center; padding: 4rem 1rem; color: #6b6b8a;">
            <div style="font-size: 3rem; margin-bottom: 1rem">🌐</div>
            <div style="font-family: 'Syne', sans-serif; font-size: 1.1rem; margin-bottom: 0.5rem">Ready to Detect</div>
            <div style="font-size: 0.85rem; line-height: 1.6">Enter text on the left and click<br>
            <strong style="color:#9c88ff">Detect &amp; Translate</strong></div>
        </div>
        """, unsafe_allow_html=True)

    elif result == "empty":
        st.warning("⚠️ Please enter some text first.")

    elif result.status == "Unable to Detect" or result.code is None:
        st.warning(
            "⚠️ Could not reliably detect a language for this input. "
            "The text may be too short, ambiguous, or contain no linguistic content "
            "(only numbers, symbols, URLs, or emoji). Try entering a longer, "
            "more descriptive sentence."
        )
        if result != "empty":
            with st.expander("Why couldn't this be detected?"):
                for n in result.notes:
                    st.caption(f"• {n}")

    else:
        badge_class = "method-badge" if "Fallback" not in result.method or "outside" not in result.method else "method-badge"
        icon = "✅" if result.status == "High Confidence" else ("⚠️" if result.status != "Low Confidence" else "❗")
        st.markdown(f'<div class="method-badge">{icon} {html.escape(result.method)}</div>', unsafe_allow_html=True)

        if result.mixed_warning:
            secondary_names = ", ".join(result.script.get("secondary", []))
            st.markdown(
                f'<div class="mixed-warning">⚠️ This input appears to mix multiple scripts '
                f'({html.escape(result.script.get("dominant",""))} + {html.escape(secondary_names)}). '
                f'The language shown below is the dominant one detected — the text may not be monolingual.</div>',
                unsafe_allow_html=True,
            )

        st.markdown(f"""
        <div class="result-detected">
            <div class="result-label">Detected Language</div>
            <div class="result-value">{flag_for(result.code)} {html.escape(result.name)}</div>
            <div class="result-sub">Code: {html.escape(result.code)} &nbsp;·&nbsp; Confidence: {result.confidence_pct:.1f}% ({result.status})</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label">📈 Relative Model Scores (Top 5)</div>', unsafe_allow_html=True)
        st.caption("These are the NB model's own class probabilities — a useful ranking signal, not independently calibrated confidence.")
        st.markdown(render_probability_bars(result.nb.top5), unsafe_allow_html=True)

        with st.expander("🔍 Model agreement (transparency)"):
            nb_name = display_name(result.nb.code) if result.nb.code else "—"
            ft_name = display_name(result.ft.code) if result.ft.code else "—"
            st.markdown(f"""
            <div class="transparency-card">
                <div class="transparency-row"><span>NB Prediction</span><b>{html.escape(nb_name)} ({result.nb.top1*100:.1f}%, {'informative' if result.nb.informative else 'no signal'})</b></div>
                <div class="transparency-row"><span>FastText Prediction</span><b>{html.escape(ft_name)} ({result.ft.confidence*100:.1f}%)</b></div>
                <div class="transparency-row"><span>Agreement</span><b>{'Yes' if result.agree else 'No'}</b></div>
                <div class="transparency-row"><span>Dominant Script</span><b>{html.escape(result.script.get('dominant') or 'n/a')}</b></div>
            </div>
            """, unsafe_allow_html=True)
            if result.notes:
                st.caption(" · ".join(result.notes))

        st.markdown("<br>", unsafe_allow_html=True)

        translation = st.session_state.translation
        if translation:
            translated_text, tstatus = translation
            if tstatus == "same":
                st.info(f"ℹ️ Text is already in {target_lang}.")
            elif tstatus == "unavailable":
                st.warning(f"⚠️ {translated_text}")
            else:
                st.markdown(f"""
                <div class="result-translated">
                    <div class="result-label">Translated to {html.escape(target_lang)}</div>
                    <div class="result-value">{html.escape(translated_text)}</div>
                    <div class="result-sub">{len(translated_text)} characters</div>
                </div>
                """, unsafe_allow_html=True)
                render_copy_button(translated_text, key="translation")

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# HOW IT WORKS
# ─────────────────────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; margin-bottom: 1.5rem">
    <div class="hero-badge">🔬 Under The Hood</div>
    <div style="font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 700; margin-top: 0.5rem">How It Works</div>
</div>
<div class="info-grid">
    <div class="info-card">
        <div class="info-icon">🧹</div>
        <div class="info-title">Script-Aware Preprocessing</div>
        <div class="info-desc">Custom TextCleaner + Unicode script analysis. URLs, emails, and digits are stripped — every script (Devanagari, CJK, Arabic, etc.) is preserved.</div>
    </div>
    <div class="info-card">
        <div class="info-icon">🤖</div>
        <div class="info-title">Naive Bayes + FastText</div>
        <div class="info-desc">A custom TF-IDF + Naive Bayes model (22 languages, 97.57% test accuracy) is cross-checked against FastText (176 languages) — never trusted alone.</div>
    </div>
    <div class="info-card">
        <div class="info-icon">🌍</div>
        <div class="info-title">Helsinki-NLP Translation</div>
        <div class="info-desc">MarianMT models translate between language pairs with a 3-level fallback (direct → tc-big → via-English), cached per language pair.</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="glass-card">
    <div class="section-label">⚙️ ML Pipeline Flow</div>
    <div style="display:flex; align-items:center; gap:0.5rem; flex-wrap:wrap; margin-top:0.8rem">
        <div style="background:#6c63ff22; border:1px solid #6c63ff44; border-radius:8px; padding:6px 14px; font-size:0.85rem">📝 Raw Text</div>
        <div style="color:#6b6b8a">→</div>
        <div style="background:#6c63ff22; border:1px solid #6c63ff44; border-radius:8px; padding:6px 14px; font-size:0.85rem">🔎 Script Analysis</div>
        <div style="color:#6b6b8a">→</div>
        <div style="background:#6c63ff22; border:1px solid #6c63ff44; border-radius:8px; padding:6px 14px; font-size:0.85rem">🤖 Naive Bayes</div>
        <div style="color:#6b6b8a">→</div>
        <div style="background:#ff658422; border:1px solid #ff658444; border-radius:8px; padding:6px 14px; font-size:0.85rem">⚡ FastText Verify</div>
        <div style="color:#6b6b8a">→</div>
        <div style="background:#ff658422; border:1px solid #ff658444; border-radius:8px; padding:6px 14px; font-size:0.85rem">🧮 Confidence Fusion</div>
        <div style="color:#6b6b8a">→</div>
        <div style="background:#43e97b22; border:1px solid #43e97b44; border-radius:8px; padding:6px 14px; font-size:0.85rem">🌍 Helsinki NLP</div>
        <div style="color:#6b6b8a">→</div>
        <div style="background:#43e97b22; border:1px solid #43e97b44; border-radius:8px; padding:6px 14px; font-size:0.85rem">✅ Output</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<br>
<div style="text-align:center; padding: 1rem 0; color: #6b6b8a; font-size: 0.8rem;">
    Built with ❤️ using Scikit-Learn · FastText · Helsinki-NLP · Streamlit
</div>
""", unsafe_allow_html=True)
