"""
TTS.ai API integration with emotion-aware voice synthesis.

Uses Kokoro model with:
 - instruct_text for emotion control
 - af_bella (female) / am_adam (male) voice selection
"""

import os
import time
import requests


# ---------- Emotion → TTS instruction mapping ---------- #
EMOTION_INSTRUCTIONS = {
    "HAPPY":     "Speak with pure joy and excitement, upbeat and fast paced.",
    "SAD":       "Speak slowly and softly, with deep sadness and grief in your voice.",
    "ANGRY":     "Speak with intense anger and frustration, sharp and aggressive tone.",
    "FEARFUL":   "Speak with trembling fear and anxiety, hesitant and shaky.",
    "SURPRISED": "Speak with great surprise and astonishment, wide-eyed excitement.",
    "DISGUSTED": "Speak with disgust and contempt, dismissive tone.",
    "NEUTRAL":   "Speak in a calm, flat, professional and neutral tone.",
}

# Voice mapping
VOICE_MAP = {
    "FEMALE": "af_bella",
    "MALE":   "am_adam",
}


class EmotionTTS:
    """Synthesize speech with emotion using the TTS.ai API (Kokoro model)."""

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str = "https://api.tts.ai/v1",
        model: str = "kokoro",
        female_voice: str = "af_bella",
        male_voice: str = "am_adam",
    ):
        self.api_key = api_key or os.getenv("TTS_API_KEY", "")
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.voices = {"FEMALE": female_voice, "MALE": male_voice}

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def synthesize(
        self,
        text: str,
        emotion: str = "NEUTRAL",
        gender: str = "FEMALE",
        output_path: str = "output.wav",
        max_polls: int = 30,
        poll_interval: float = 2.0,
    ) -> str:
        """
        Submit TTS job → poll for result → save audio file.

        Returns the path to the saved audio file.
        """
        emotion = emotion.upper()
        gender = gender.upper()

        voice = self.voices.get(gender, self.voices["FEMALE"])
        instruct = EMOTION_INSTRUCTIONS.get(emotion, EMOTION_INSTRUCTIONS["NEUTRAL"])

        print(f"[TTS] Submitting: emotion={emotion}, gender={gender}, voice={voice}")
        print(f"[TTS] Text: {text[:80]}{'...' if len(text) > 80 else ''}")

        # --- Submit job ---
        resp = requests.post(
            f"{self.api_base}/tts/",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "text": text,
                "voice": voice,
                "instruct_text": instruct,
                "format": "wav",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        uuid = data.get("uuid")
        print(f"[TTS] Queued — UUID: {uuid}")

        # --- Poll for result ---
        for attempt in range(max_polls):
            time.sleep(poll_interval)
            result = requests.get(
                f"{self.api_base}/speech/results/",
                params={"uuid": uuid},
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30,
            ).json()

            status = result.get("status", "unknown")
            print(f"[TTS]   Polling #{attempt + 1}: {status}")

            if status == "completed":
                audio_url = result.get("result_url")
                audio_data = requests.get(audio_url, timeout=60).content
                with open(output_path, "wb") as f:
                    f.write(audio_data)
                print(f"[TTS] ✅ Saved → {output_path}")
                return output_path

            elif status == "failed":
                raise RuntimeError(f"TTS job failed: {result}")

        raise TimeoutError(f"TTS job timed out after {max_polls * poll_interval}s")


# ------------------------------------------------------------------ #
#  Quick self-test                                                     #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    tts = EmotionTTS()
    tts.synthesize(
        text="Hello! I am so excited to be speaking to you today!",
        emotion="HAPPY",
        gender="FEMALE",
        output_path="test_happy.wav",
    )
