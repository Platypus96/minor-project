"""
SenseVoice-based Speech-to-Text with Emotion and Gender detection.

Uses:
- FunASR SenseVoice-Small for STT + emotion extraction
- Pitch analysis (F0) for gender detection
"""

import re
import numpy as np
import torch
import torchaudio


class SpeechEmotionExtractor:
    """Extracts text, emotion, and speaker gender from audio."""

    def __init__(self, device: str = "cpu"):
        from funasr import AutoModel

        self.device = device
        print(f"[SenseVoice] Loading model on {device}...")
        self.model = AutoModel(
            model="iic/SenseVoiceSmall",
            trust_remote_code=True,
            device=device,
        )
        print("[SenseVoice] Model loaded.")

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def extract(self, audio_path: str, language: str = "auto") -> dict:
        """
        Analyse an audio file and return::

            {
                "text":          str,  # cleaned transcript
                "emotion":       str,  # HAPPY | SAD | ANGRY | NEUTRAL | ...
                "gender":        str,  # MALE | FEMALE
                "detected_lang": str,  # ISO 639-1 code, e.g. 'en', 'zh'
                "raw":           str,  # raw model output (for debugging)
            }

        Parameters
        ----------
        language : str
            ISO 639-1 code ('en', 'zh', 'hi', ...) or 'auto' for
            SenseVoice to detect the language automatically.
        """
        from pipeline.translation import detect_language_from_tag

        # SenseVoice expects 'auto' or a specific language code
        sv_lang = language if language != "auto" else "auto"

        # --- SenseVoice inference ---
        result = self.model.generate(
            input=audio_path,
            cache={},
            language=sv_lang,
            use_itn=True,
            batch_size_s=60,
        )
        raw_text = result[0]["text"] if result else ""

        # --- Parse emotion & clean text ---
        emotion = self._parse_emotion(raw_text)
        clean_text = self._clean_text(raw_text)

        # --- Detect language from SenseVoice tags ---
        detected_lang = detect_language_from_tag(raw_text) if language == "auto" else language

        # --- Gender from pitch ---
        gender = self._detect_gender(audio_path)

        return {
            "text": clean_text,
            "emotion": emotion,
            "gender": gender,
            "detected_lang": detected_lang,
            "raw": raw_text,
        }

    # ------------------------------------------------------------------ #
    #  Internals                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_emotion(raw: str) -> str:
        """
        SenseVoice output format includes tags like:
        <|en|><|HAPPY|><|Speech|><|woitn|>actual text here

        We extract the emotion tag.
        """
        # Known emotion tags from SenseVoice
        known_emotions = {"HAPPY", "SAD", "ANGRY", "FEARFUL", "DISGUSTED", "SURPRISED", "NEUTRAL"}
        matches = re.findall(r"<\|(\w+)\|>", raw)
        for m in matches:
            if m.upper() in known_emotions:
                return m.upper()
        return "NEUTRAL"

    @staticmethod
    def _clean_text(raw: str) -> str:
        """Remove all <|TAG|> markers and return clean text."""
        cleaned = re.sub(r"<\|[^|]*\|>", "", raw).strip()
        return cleaned if cleaned else raw.strip()

    @staticmethod
    def _detect_gender(audio_path: str) -> str:
        """
        Estimate speaker gender from fundamental frequency (F0).

        Typical ranges:
          Male   → 85 – 180 Hz  (avg ~120 Hz)
          Female → 165 – 255 Hz (avg ~210 Hz)

        Threshold: 165 Hz
        """
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Use torchaudio's pitch detection
            pitch = torchaudio.functional.detect_pitch_frequency(
                waveform, sample_rate
            )

            # Filter out silence / unvoiced (pitch == 0)
            voiced = pitch[pitch > 50]
            if len(voiced) == 0:
                return "NEUTRAL_GENDER"

            median_f0 = voiced.median().item()
            return "FEMALE" if median_f0 > 165 else "MALE"

        except Exception as e:
            print(f"[Gender] Pitch detection failed: {e}, defaulting to FEMALE")
            return "FEMALE"


# ------------------------------------------------------------------ #
#  Quick self-test                                                     #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "sample.wav"
    extractor = SpeechEmotionExtractor(device="cpu")
    result = extractor.extract(path)
    print(f"\n{'='*50}")
    print(f"Text   : {result['text']}")
    print(f"Emotion: {result['emotion']}")
    print(f"Gender : {result['gender']}")
    print(f"Raw    : {result['raw']}")
    print(f"{'='*50}")
