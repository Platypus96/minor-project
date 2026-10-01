"""
Test Audio Corpus Manager for Benchmarking.

Manages the standardized set of audio samples used in the benchmark.
Handles loading, validation, duration measurement, and metadata
for the test corpus.

The corpus is designed to cover a range of utterance lengths
so we can measure how latency scales with audio duration.
"""

import os
import sys
import wave
from dataclasses import dataclass

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pipeline.benchmark.timing import get_audio_duration


@dataclass
class AudioSample:
    """Metadata for a single test audio sample."""
    path: str
    filename: str
    duration_s: float
    category: str           # "short", "medium", "long"
    language: str            # ISO 639-1 code
    expected_text: str       # Ground truth transcript (if known)
    exists: bool = True


# ------------------------------------------------------------------ #
#  Corpus definition                                                   #
# ------------------------------------------------------------------ #

# These are the standard samples to use in the benchmark.
# The corpus covers three duration categories:
#   short  : 1-3 seconds  (single phrase)
#   medium : 3-10 seconds (one or two sentences)
#   long   : 10-30 seconds (paragraph-level)

CORPUS_DEFINITION = [
    # --- Existing samples ---
    {
        "filename": "english_hello_how_are_you.wav",
        "category": "short",
        "language": "en",
        "expected_text": "Hello, how are you?",
    },
    {
        "filename": "chinese_wo_ai_zhongguo.wav",
        "category": "short",
        "language": "zh",
        "expected_text": "我爱中国",
    },
    # --- Samples to generate (see generate_corpus() below) ---
    {
        "filename": "en_short_greeting.wav",
        "category": "short",
        "language": "en",
        "expected_text": "Good morning, how are you doing today?",
    },
    {
        "filename": "en_medium_sentence.wav",
        "category": "medium",
        "language": "en",
        "expected_text": "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the English alphabet.",
    },
    {
        "filename": "en_medium_technical.wav",
        "category": "medium",
        "language": "en",
        "expected_text": "Semantic communication focuses on transmitting the meaning of information rather than exact bit sequences. This approach significantly reduces bandwidth requirements.",
    },
    {
        "filename": "en_long_paragraph.wav",
        "category": "long",
        "language": "en",
        "expected_text": (
            "In modern wireless communication systems, the exponential growth of data traffic "
            "has pushed traditional Shannon capacity limits. Semantic communication offers a "
            "paradigm shift by focusing on meaning extraction and transmission. Deep learning "
            "models like DeepSC use transformer architectures to encode semantic features into "
            "compact representations that are robust to channel noise."
        ),
    },
]


class TestCorpus:
    """
    Manages the benchmark test audio corpus.

    Scans the test_audio/ directory, validates files, measures durations,
    and provides an iterator for benchmark scripts.
    """

    def __init__(self, corpus_dir: str | None = None):
        self.corpus_dir = corpus_dir or os.path.join(_PROJECT_ROOT, "test_audio")
        self.samples: list[AudioSample] = []
        self._scan()

    def _scan(self):
        """Scan the corpus directory and load metadata for available files."""
        self.samples = []

        for entry in CORPUS_DEFINITION:
            path = os.path.join(self.corpus_dir, entry["filename"])
            exists = os.path.isfile(path)

            duration = get_audio_duration(path) if exists else 0.0

            sample = AudioSample(
                path=path,
                filename=entry["filename"],
                duration_s=duration,
                category=entry["category"],
                language=entry["language"],
                expected_text=entry["expected_text"],
                exists=exists,
            )
            self.samples.append(sample)

    def get_available(self, language: str | None = None, category: str | None = None) -> list[AudioSample]:
        """
        Get available (existing) audio samples, optionally filtered.

        Parameters
        ----------
        language : str, optional
            Filter by language code (e.g., 'en').
        category : str, optional
            Filter by duration category ('short', 'medium', 'long').

        Returns
        -------
        list[AudioSample]
            Available audio samples matching the filters.
        """
        result = [s for s in self.samples if s.exists]
        if language:
            result = [s for s in result if s.language == language]
        if category:
            result = [s for s in result if s.category == category]
        return result

    def get_missing(self) -> list[AudioSample]:
        """Get samples that are defined but missing from disk."""
        return [s for s in self.samples if not s.exists]

    def print_status(self):
        """Print a status report of the corpus."""
        print(f"\n{'='*70}")
        print(f"  TEST AUDIO CORPUS -- {self.corpus_dir}")
        print(f"{'='*70}")
        print(f"  {'Status':<8} {'File':<35} {'Cat':<8} {'Lang':<6} {'Duration':<10}")
        print(f"  {'-'*8} {'-'*35} {'-'*8} {'-'*6} {'-'*10}")

        for s in self.samples:
            status = "  [Y]" if s.exists else "  [N]"
            dur = f"{s.duration_s:.2f}s" if s.exists else "N/A"
            print(f"  {status:<8} {s.filename:<35} {s.category:<8} {s.language:<6} {dur:<10}")

        available = len(self.get_available())
        missing = len(self.get_missing())
        print(f"\n  Available: {available} / {len(self.samples)}  |  Missing: {missing}")

        if missing > 0:
            print(f"\n  TIP: Run 'python corpus.py --generate' to create missing samples")
            print(f"     using TTS, or record them manually.")
        print(f"{'='*70}\n")

    def generate_missing_with_tts(self, tts_engine=None):
        """
        Generate missing corpus audio files using TTS.

        Parameters
        ----------
        tts_engine : EmotionTTS, optional
            TTS engine instance. If None, uses a simple approach with
            pyttsx3 (offline) or falls back to creating silent placeholders.
        """
        missing = self.get_missing()
        if not missing:
            print("[Corpus] All samples exist. Nothing to generate.")
            return

        os.makedirs(self.corpus_dir, exist_ok=True)

        for sample in missing:
            if sample.language != "en":
                print(f"[Corpus] Skipping {sample.filename} (non-English, record manually)")
                continue

            print(f"[Corpus] Generating: {sample.filename}")
            print(f"  Text: {sample.expected_text[:60]}...")

            # Try pyttsx3 (offline TTS) first
            generated = False
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty("rate", 150)  # Moderate speed
                engine.save_to_file(sample.expected_text, sample.path)
                engine.runAndWait()
                generated = True
                print(f"  [OK] Generated with pyttsx3")
            except Exception as e:
                print(f"  WARN: pyttsx3 failed: {e}")

            # Fallback: create a placeholder WAV with silence
            if not generated:
                try:
                    self._create_silent_wav(
                        sample.path,
                        duration_s=len(sample.expected_text.split()) * 0.4,  # ~0.4s per word
                    )
                    generated = True
                    print(f"  WARN: Created silent placeholder (record real audio for accuracy)")
                except Exception as e:
                    print(f"  FAIL: Failed to create placeholder: {e}")

        # Re-scan to update metadata
        self._scan()

    @staticmethod
    def _create_silent_wav(path: str, duration_s: float = 3.0, sample_rate: int = 16000):
        """Create a silent WAV file as a placeholder."""
        import struct
        num_samples = int(duration_s * sample_rate)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(struct.pack(f"<{num_samples}h", *([0] * num_samples)))


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import sys

    corpus = TestCorpus()
    corpus.print_status()

    if "--generate" in sys.argv:
        corpus.generate_missing_with_tts()
        corpus.print_status()
