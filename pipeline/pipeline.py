"""
End-to-end Semantic Speech Communication Pipeline.

Flow:
  1. Speech  → SenseVoice → text + emotion + gender
  2. text    → DeepSC encoder → channel → DeepSC decoder → reconstructed text
  3. text + emotion + gender → TTS API → output speech

Emotion and gender are side-channel metadata (not passed through DeepSC).
"""

import os
import sys
from dotenv import load_dotenv

# Add project root to path so imports work when running from any directory
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pipeline.stt_emotion.sense_voice import SpeechEmotionExtractor
from pipeline.semantic.deepsc_wrapper import DeepSCWrapper
from pipeline.tts.tts_api import EmotionTTS
from pipeline.translation import translate_to_english, translate_from_english
from pipeline.semantic.constellation import plot_constellation


class SemanticSpeechPipeline:
    """Orchestrates the full speech → semantic comm → speech pipeline."""

    def __init__(self, config: dict | None = None):
        load_dotenv()
        config = config or {}

        device = config.get("device", os.getenv("DEVICE", "cpu"))

        # --- 1. STT + Emotion + Gender ---
        print("=" * 60)
        print("  Initializing Semantic Speech Pipeline")
        print("=" * 60)
        self.stt = SpeechEmotionExtractor(device=device)

        # --- 2. DeepSC (semantic encoder/decoder + channel) ---
        # Resolve checkpoint directory
        checkpoint_dir = config.get(
            "checkpoint_dir",
            os.getenv("CHECKPOINT_DIR", os.path.join(_PROJECT_ROOT, "checkpoints")),
        )
        vocab_path = config.get(
            "vocab_path",
            os.getenv("VOCAB_PATH", os.path.join(checkpoint_dir, "vocab.json")),
        )

        self.deepsc = DeepSCWrapper(
            checkpoint_dir=checkpoint_dir,
            checkpoint_path=config.get(
                "checkpoint_path",
                os.getenv("CHECKPOINT_PATH", None),
            ),
            vocab_path=vocab_path,
            channel_type=config.get("channel", os.getenv("CHANNEL", "AWGN")),
            snr_db=float(config.get("snr_db", os.getenv("SNR_DB", "10.0"))),
            device=device,
        )

        # --- 3. TTS ---
        self.tts = EmotionTTS(
            api_key=config.get("tts_api_key", os.getenv("TTS_API_KEY", "")),
            api_base=config.get("tts_api_base", os.getenv("TTS_API_BASE", "https://api.tts.ai/v1")),
            model=config.get("tts_model", os.getenv("TTS_MODEL", "kokoro")),
            female_voice=config.get("female_voice", os.getenv("FEMALE_VOICE", "af_bella")),
            male_voice=config.get("male_voice", os.getenv("MALE_VOICE", "am_adam")),
        )

        print("=" * 60)
        print(f"  Pipeline ready  |  DeepSC: {self.deepsc}")
        print("=" * 60)

    # ------------------------------------------------------------------ #
    #  Main entry point                                                    #
    # ------------------------------------------------------------------ #

    def run(
        self,
        audio_input_path: str,
        output_path: str = "output.wav",
        channel: str | None = None,
        snr_db: float | None = None,
        language: str = "auto",
    ) -> dict:
        """
        Run the full pipeline on an input audio file.

        Parameters
        ----------
        language : str
            'auto' to detect language automatically, or an ISO 639-1 code
            ('en', 'hi', 'zh', 'ja', ...).  Non-English audio is translated
            to English before DeepSC and back to the source language after.

        Returns a dict with all intermediate results.
        """
        # Update channel if overridden
        if channel or snr_db is not None:
            self.deepsc.set_channel(
                channel_type=channel or self.deepsc.channel.channel_type,
                snr_db=snr_db if snr_db is not None else self.deepsc.channel.snr_db,
            )

        # ---- Step 1: Speech → Text + Emotion + Gender ----
        print("\n[Step 1] Speech-to-Text + Emotion + Gender")
        extracted = self.stt.extract(audio_input_path, language=language)
        text = extracted["text"]
        emotion = extracted["emotion"]
        gender = extracted["gender"]
        detected_lang = extracted.get("detected_lang", "en")
        print(f"   Text   : {text}")
        print(f"   Emotion: {emotion}")
        print(f"   Gender : {gender}")
        print(f"   Lang   : {detected_lang}")

        # ---- Step 1b: Translate to English if needed ----
        text_en = translate_to_english(text, detected_lang)
        if text_en != text:
            print(f"   Translated (en): {text_en}")

        # ---- Step 2: Semantic Communication (always in English) ----
        print(f"\n[Step 2] Semantic Communication ({self.deepsc.channel})")
        reconstructed_en = self.deepsc.transmit(text_en)
        print(f"   Reconstructed (en): {reconstructed_en}")

        # ---- Step 2b: Translate reconstructed text back to source language ----
        reconstructed_text = translate_from_english(reconstructed_en, detected_lang)
        if reconstructed_text != reconstructed_en:
            print(f"   Reconstructed ({detected_lang}): {reconstructed_text}")

        # ---- Step 3: Compute text similarity (compare English sides) ----
        bleu = self._compute_bleu(text_en, reconstructed_en)
        print(f"   BLEU score: {bleu:.4f}")

        # ---- Step 3b: Generate constellation diagram ----
        constellation_path = None
        try:
            pre_sym, post_sym = self.deepsc.get_last_symbols()
            if pre_sym is not None and post_sym is not None:
                const_dir = os.path.join(_PROJECT_ROOT, "outputs")
                os.makedirs(const_dir, exist_ok=True)
                constellation_path = plot_constellation(
                    pre_sym, post_sym,
                    channel_type=self.deepsc.channel.channel_type,
                    snr_db=self.deepsc.channel.snr_db,
                    output_path=os.path.join(const_dir, "constellation.png"),
                )
        except Exception as e:
            print(f"[Constellation] Warning: {e}")

        # ---- Step 4: TTS with emotion ----
        print(f"\n[Step 3] TTS (emotion={emotion}, gender={gender})")
        audio_out = self.tts.synthesize(
            text=reconstructed_text,
            emotion=emotion,
            gender=gender,
            output_path=output_path,
        )

        return {
            "original_text": text,
            "original_text_en": text_en,
            "emotion": emotion,
            "gender": gender,
            "detected_lang": detected_lang,
            "reconstructed_text": reconstructed_text,
            "reconstructed_text_en": reconstructed_en,
            "bleu_score": bleu,
            "channel": self.deepsc.channel.channel_type,
            "snr_db": self.deepsc.channel.snr_db,
            "output_audio": audio_out,
            "deepsc_mode": "MOCK" if self.deepsc.mock_mode else "REAL",
            "constellation_path": constellation_path,
        }

    def _compute_bleu(self, text: str, reconstructed_text: str) -> float:
        """Compute BLEU score between original and reconstructed text.

        Both sides are normalized using DeepSC's own preprocessing so the
        comparison is fair -- scoring semantic fidelity, not surface
        differences like capitalisation or punctuation that DeepSC strips
        internally before encoding.
        """
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from pipeline.semantic.deepsc_wrapper import _normalize_string
        try:
            smoothie = SmoothingFunction().method4
            ref_tokens = _normalize_string(text).split()
            hyp_tokens = _normalize_string(reconstructed_text).split()
            if not ref_tokens or not hyp_tokens:
                return 0.0
                
            # For longer texts in semantic comms, 4-gram matching is overly harsh
            # if the model uses synonyms or drops articles. We scale weights based on length.
            length = len(ref_tokens)
            if length == 1:
                weights = (1.0,)
            elif length == 2:
                weights = (0.5, 0.5)
            elif length == 3:
                weights = (0.33, 0.33, 0.33)
            else:
                # Use BLEU-3 effectively for longer text to be more forgiving
                weights = (0.4, 0.3, 0.3, 0.0)

            return sentence_bleu(
                [ref_tokens],
                hyp_tokens,
                weights=weights,
                smoothing_function=smoothie,
            )
        except Exception:
            return 0.0

    def run_streaming(
        self,
        audio_input_path: str,
        output_path: str = "output.wav",
        channel: str | None = None,
        snr_db: float | None = None,
        language: str = "auto",
    ):
        """
        Generator that yields intermediate state after each pipeline step.
        Used by the Gradio UI for step-by-step progressive updates.
        """
        # Update channel if overridden
        if channel or snr_db is not None:
            self.deepsc.set_channel(
                channel_type=channel or self.deepsc.channel.channel_type,
                snr_db=snr_db if snr_db is not None else self.deepsc.channel.snr_db,
            )

        # Step 1: STT
        yield {"step": 1, "status": "running", "label": "Speech Recognition"}
        extracted = self.stt.extract(audio_input_path, language=language)
        detected_lang = extracted.get("detected_lang", "en")
        text_en = translate_to_english(extracted["text"], detected_lang)
        yield {
            "step": 1, "status": "done",
            "text": extracted["text"],
            "text_en": text_en,
            "emotion": extracted["emotion"],
            "gender": extracted["gender"],
            "detected_lang": detected_lang,
        }

        # Step 2: Semantic Communication
        yield {"step": 2, "status": "running", "label": "Semantic Encoding + Channel"}
        reconstructed_en = self.deepsc.transmit(text_en)
        reconstructed_text = translate_from_english(reconstructed_en, detected_lang)
        bleu = self._compute_bleu(text_en, reconstructed_en)

        # Generate constellation diagram
        constellation_path = None
        try:
            pre_sym, post_sym = self.deepsc.get_last_symbols()
            if pre_sym is not None and post_sym is not None:
                const_dir = os.path.join(_PROJECT_ROOT, "outputs")
                os.makedirs(const_dir, exist_ok=True)
                constellation_path = plot_constellation(
                    pre_sym, post_sym,
                    channel_type=self.deepsc.channel.channel_type,
                    snr_db=self.deepsc.channel.snr_db,
                    output_path=os.path.join(const_dir, "constellation.png"),
                )
        except Exception as e:
            print(f"[Constellation] Warning: {e}")

        yield {
            "step": 2, "status": "done",
            "original_text": extracted["text"],
            "original_text_en": text_en,
            "reconstructed_text": reconstructed_text,
            "reconstructed_text_en": reconstructed_en,
            "bleu_score": bleu,
            "channel": self.deepsc.channel.channel_type,
            "snr_db": self.deepsc.channel.snr_db,
            "deepsc_mode": "MOCK" if self.deepsc.mock_mode else "REAL",
            "constellation_path": constellation_path,
        }

        # Step 3: TTS
        yield {"step": 3, "status": "running", "label": "Speech Synthesis"}
        audio_out = self.tts.synthesize(
            text=reconstructed_text,
            emotion=extracted["emotion"],
            gender=extracted["gender"],
            output_path=output_path,
        )
        yield {
            "step": 3, "status": "done",
            "output_audio": audio_out,
        }


# ------------------------------------------------------------------ #
#  CLI entry point                                                     #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <audio_file.wav>")
        sys.exit(1)

    pipe = SemanticSpeechPipeline()
    result = pipe.run(sys.argv[1])
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    for k, v in result.items():
        print(f"  {k}: {v}")
