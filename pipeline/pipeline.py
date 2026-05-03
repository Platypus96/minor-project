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
    ) -> dict:
        """
        Run the full pipeline on an input audio file.

        Returns a dict with all intermediate results.
        """
        # Update channel if overridden
        if channel or snr_db is not None:
            self.deepsc.set_channel(
                channel_type=channel or self.deepsc.channel.channel_type,
                snr_db=snr_db if snr_db is not None else self.deepsc.channel.snr_db,
            )

        # ---- Step 1: Speech → Text + Emotion + Gender ----
        print("\n🎤 Step 1: Speech-to-Text + Emotion + Gender")
        extracted = self.stt.extract(audio_input_path)
        text = extracted["text"]
        emotion = extracted["emotion"]
        gender = extracted["gender"]
        print(f"   📝 Text   : {text}")
        print(f"   😊 Emotion: {emotion}")
        print(f"   👤 Gender : {gender}")

        # ---- Step 2: Semantic Communication ----
        print(f"\n📡 Step 2: Semantic Communication ({self.deepsc.channel})")
        reconstructed_text = self.deepsc.transmit(text)
        print(f"   🔁 Reconstructed: {reconstructed_text}")

        # ---- Step 3: Compute text similarity ----
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        try:
            smoothie = SmoothingFunction().method1
            bleu = sentence_bleu(
                [text.lower().split()],
                reconstructed_text.lower().split(),
                smoothing_function=smoothie,
            )
        except Exception:
            bleu = 0.0
        print(f"   📊 BLEU score: {bleu:.4f}")

        # ---- Step 4: TTS with emotion ----
        print(f"\n🔊 Step 3: TTS (emotion={emotion}, gender={gender})")
        audio_out = self.tts.synthesize(
            text=reconstructed_text,
            emotion=emotion,
            gender=gender,
            output_path=output_path,
        )

        return {
            "original_text": text,
            "emotion": emotion,
            "gender": gender,
            "reconstructed_text": reconstructed_text,
            "bleu_score": bleu,
            "channel": self.deepsc.channel.channel_type,
            "snr_db": self.deepsc.channel.snr_db,
            "output_audio": audio_out,
            "deepsc_mode": "MOCK" if self.deepsc.mock_mode else "REAL",
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
