"""
Translation bridge for multilingual support.

DeepSC is trained on English-only text. For non-English audio:
  1. SenseVoice transcribes in the source language
  2. We translate to English before DeepSC encoding
  3. After DeepSC decoding, we translate the English result back
     to the source language for TTS

Uses `deep_translator` (Google Translate backend, no API key required).

Install:  pip install deep-translator
"""

SUPPORTED_LANGUAGES = {
    "auto":  "Auto-detect",
    "en":    "English",
    "hi":    "Hindi",
    "zh":    "Chinese (Simplified)",
    "ja":    "Japanese",
    "ko":    "Korean",
    "fr":    "French",
    "de":    "German",
    "es":    "Spanish",
    "ar":    "Arabic",
    "pt":    "Portuguese",
    "ru":    "Russian",
    "it":    "Italian",
}

# SenseVoice language tag -> ISO 639-1 code
SENSEVOICE_LANG_MAP = {
    "en":   "en",
    "zh":   "zh",
    "ja":   "ja",
    "ko":   "ko",
    "yue":  "zh",  # Cantonese -> treat as Chinese for translation
}

# Emoji-free language display labels for the UI dropdown
LANG_CHOICES = [
    ("Auto-detect",          "auto"),
    ("English",              "en"),
    ("Hindi",                "hi"),
    ("Chinese (Simplified)", "zh"),
    ("Japanese",             "ja"),
    ("Korean",               "ko"),
    ("French",               "fr"),
    ("German",               "de"),
    ("Spanish",              "es"),
    ("Arabic",               "ar"),
    ("Portuguese",           "pt"),
    ("Russian",              "ru"),
    ("Italian",              "it"),
]


def _get_translator():
    """Import deep_translator lazily with a helpful error message."""
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator
    except ImportError:
        raise ImportError(
            "deep-translator is not installed. Run:\n"
            "  pip install deep-translator\n"
            "or add it to requirements.txt and re-run install.bat"
        )


def detect_language_from_tag(raw_sensevoice_output: str) -> str:
    """
    Extract the language tag from SenseVoice's raw output.

    SenseVoice prefixes output with tags like:
        <|en|><|NEUTRAL|><|Speech|>...
        <|zh|><|HAPPY|><|Speech|>...

    Returns an ISO code like 'en', 'zh', 'hi', or 'en' as fallback.
    """
    import re
    tags = re.findall(r"<\|(\w+)\|>", raw_sensevoice_output)
    for tag in tags:
        tag_lower = tag.lower()
        if tag_lower in SENSEVOICE_LANG_MAP:
            return SENSEVOICE_LANG_MAP[tag_lower]
        # SenseVoice sometimes emits full ISO codes for languages it detects
        if len(tag_lower) == 2 and tag_lower in SUPPORTED_LANGUAGES:
            return tag_lower
    return "en"


def translate_to_english(text: str, source_lang: str) -> str:
    """
    Translate `text` from `source_lang` to English.
    Returns the original text unchanged if source_lang is 'en' or translation
    fails.
    """
    if source_lang == "en" or not text.strip():
        return text
    try:
        GoogleTranslator = _get_translator()
        translated = GoogleTranslator(source=source_lang, target="en").translate(text)
        print(f"[Translate] {source_lang} -> en: '{text[:60]}...' => '{translated[:60]}...'")
        return translated or text
    except Exception as e:
        print(f"[Translate] Warning: translation failed ({e}), using original text.")
        return text


def translate_from_english(text: str, target_lang: str) -> str:
    """
    Translate `text` from English to `target_lang`.
    Returns the original text unchanged if target_lang is 'en' or translation
    fails.
    """
    if target_lang == "en" or not text.strip():
        return text
    try:
        GoogleTranslator = _get_translator()
        translated = GoogleTranslator(source="en", target=target_lang).translate(text)
        print(f"[Translate] en -> {target_lang}: '{text[:60]}...' => '{translated[:60]}...'")
        return translated or text
    except Exception as e:
        print(f"[Translate] Warning: reverse translation failed ({e}), using English.")
        return text
