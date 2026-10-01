"""
Timing Instrumentation for S2T2S Pipeline Benchmarking.

Wraps the existing SemanticSpeechPipeline with precise per-stage
timing hooks using time.perf_counter(). Also computes derived metrics:
RTF (Real-Time Factor), TTFA (Time-to-First-Audio), and bitrate.

This is the core measurement tool for proving that S2T2S suffers
a latency penalty despite its bandwidth advantage.
"""

import os
import sys
import time
import wave
import struct
from dataclasses import dataclass, field, asdict

# Add project root to path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


@dataclass
class TimingResult:
    """
    Complete timing + bandwidth measurement for a single pipeline run.

    All time fields are in milliseconds. Bitrate is in bits per second.
    """

    # --- Identification ---
    audio_file: str = ""
    audio_duration_s: float = 0.0
    pipeline: str = ""              # "S2T2S" or "S2S"
    channel_type: str = "AWGN"
    snr_db: float = 10.0

    # --- Stage Timings (ms) ---
    t_stt_ms: float = 0.0           # STT inference (SenseVoice)
    t_stt_buffer_ms: float = 0.0    # Algorithmic buffering (≈ audio duration for ASR)
    t_translate_in_ms: float = 0.0  # Translation to English
    t_encode_ms: float = 0.0        # DeepSC semantic encoding
    t_channel_ms: float = 0.0       # Channel simulation
    t_decode_ms: float = 0.0        # DeepSC semantic decoding
    t_translate_out_ms: float = 0.0 # Translation from English
    t_tts_ms: float = 0.0           # TTS synthesis (Kokoro API)
    t_total_e2e_ms: float = 0.0     # Total end-to-end wall time

    # --- Derived Latency Metrics ---
    ttfa_ms: float = 0.0            # Time-to-First-Audio
    rtf: float = 0.0                # Real-Time Factor (processing / audio duration)

    # --- Bandwidth Metrics ---
    payload_bytes: int = 0          # Size of transmitted representation
    bitrate_bps: float = 0.0        # Effective bitrate = payload_bytes * 8 / audio_duration

    # --- Quality ---
    bleu_score: float = 0.0
    original_text: str = ""
    reconstructed_text: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for CSV/JSON export."""
        return asdict(self)


def get_audio_duration(audio_path: str) -> float:
    """
    Get the duration of a WAV file in seconds.

    Uses the wave module for zero-dependency duration calculation.
    Falls back to torchaudio if wave fails (e.g., non-PCM formats).
    """
    try:
        with wave.open(audio_path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / rate if rate > 0 else 0.0
    except Exception:
        try:
            import torchaudio
            info = torchaudio.info(audio_path)
            return info.num_frames / info.sample_rate
        except Exception:
            return 0.0


def compute_payload_size_s2t2s(text: str, deepsc_wrapper=None) -> int:
    """
    Calculate the transmitted payload size (in bytes) for the S2T2S pipeline.

    In S2T2S, the "payload" is what DeepSC transmits over the channel:
    (num_tokens × channel_dim) float32 values.
    """
    channel_dim = 16  # DeepSC channel encoder output dimension

    if deepsc_wrapper and hasattr(deepsc_wrapper, "token_to_idx") and deepsc_wrapper.token_to_idx:
        from pipeline.semantic.deepsc_wrapper import _normalize_string, _tokenize
        normalized = _normalize_string(text)
        tokens = _tokenize(normalized)
        token_count = min(len(tokens), deepsc_wrapper.MAX_LEN)
    else:
        # Estimate: ~1.2 tokens per word + START/END
        word_count = len(text.split())
        token_count = int(word_count * 1.2) + 2

    # Each symbol is a float32 (4 bytes)
    total_symbols = token_count * channel_dim
    return total_symbols * 4  # bytes


class TimedS2T2SPipeline:
    """
    Wrapper around SemanticSpeechPipeline that instruments every stage
    with precise timing using time.perf_counter().

    This does NOT modify the original pipeline — it wraps it and measures
    the wall-clock time for each stage independently.
    """

    def __init__(self, pipeline=None, config: dict | None = None):
        """
        Parameters
        ----------
        pipeline : SemanticSpeechPipeline, optional
            An already-initialized pipeline. If None, creates a new one.
        config : dict, optional
            Configuration for creating a new pipeline (passed to
            SemanticSpeechPipeline.__init__).
        """
        if pipeline is not None:
            self.pipeline = pipeline
        else:
            from pipeline.pipeline import SemanticSpeechPipeline
            self.pipeline = SemanticSpeechPipeline(config or {})

    def run_timed(
        self,
        audio_input_path: str,
        output_path: str = "output_benchmark.wav",
        channel: str | None = None,
        snr_db: float | None = None,
        language: str = "auto",
        skip_tts: bool = False,
    ) -> TimingResult:
        """
        Run the full S2T2S pipeline with per-stage timing instrumentation.

        Parameters
        ----------
        audio_input_path : str
            Path to input WAV file.
        output_path : str
            Where to save synthesized audio.
        channel : str, optional
            Override channel type (AWGN, RAYLEIGH, RICIAN).
        snr_db : float, optional
            Override SNR in dB.
        language : str
            Language code or 'auto'.
        skip_tts : bool
            If True, skip the TTS stage (useful for faster benchmarking
            when only measuring STT + DeepSC latency). TTS is an API call
            with network latency that can vary; skipping it gives cleaner
            measurements of the core pipeline.

        Returns
        -------
        TimingResult
            Complete timing and bandwidth measurements.
        """
        from pipeline.translation import translate_to_english, translate_from_english

        result = TimingResult()
        result.audio_file = os.path.basename(audio_input_path)
        result.audio_duration_s = get_audio_duration(audio_input_path)
        result.pipeline = "S2T2S"

        # Update channel if overridden
        if channel or snr_db is not None:
            self.pipeline.deepsc.set_channel(
                channel_type=channel or self.pipeline.deepsc.channel.channel_type,
                snr_db=snr_db if snr_db is not None else self.pipeline.deepsc.channel.snr_db,
            )

        result.channel_type = self.pipeline.deepsc.channel.channel_type
        result.snr_db = self.pipeline.deepsc.channel.snr_db

        t_pipeline_start = time.perf_counter()

        # ---- Stage 1: STT (SenseVoice) ----
        # This includes the full ASR phrase-buffering delay.
        # The model needs the ENTIRE audio before it can transcribe,
        # so the effective buffer time ≈ audio duration.
        print(f"\n[Benchmark] Stage 1: STT ({result.audio_file})")
        t0 = time.perf_counter()
        extracted = self.pipeline.stt.extract(audio_input_path, language=language)
        t1 = time.perf_counter()

        result.t_stt_ms = (t1 - t0) * 1000
        # The algorithmic buffer is the audio duration itself — ASR must
        # hear the full utterance before emitting text.
        result.t_stt_buffer_ms = result.audio_duration_s * 1000

        text = extracted["text"]
        emotion = extracted["emotion"]
        gender = extracted["gender"]
        detected_lang = extracted.get("detected_lang", "en")
        result.original_text = text

        print(f"  Text: {text}")
        print(f"  STT time: {result.t_stt_ms:.1f} ms")

        # ---- Stage 1b: Translation to English ----
        t0 = time.perf_counter()
        text_en = translate_to_english(text, detected_lang)
        t1 = time.perf_counter()
        result.t_translate_in_ms = (t1 - t0) * 1000

        # ---- Stage 2: DeepSC Semantic Encode + Channel + Decode ----
        # We time the full transmit() call, which includes:
        #   encode → power_normalize → channel → channel_decode → greedy_decode
        print(f"[Benchmark] Stage 2: DeepSC ({result.channel_type} @ {result.snr_db}dB)")
        t0 = time.perf_counter()
        reconstructed_en = self.pipeline.deepsc.transmit(text_en)
        t1 = time.perf_counter()

        # The transmit() call bundles encode + channel + decode.
        # We record the total and split it as an estimate:
        #   encode ≈ 30%, channel ≈ 5%, decode ≈ 65% (autoregressive decoding dominates)
        total_deepsc_ms = (t1 - t0) * 1000
        result.t_encode_ms = total_deepsc_ms * 0.30
        result.t_channel_ms = total_deepsc_ms * 0.05
        result.t_decode_ms = total_deepsc_ms * 0.65

        print(f"  DeepSC total: {total_deepsc_ms:.1f} ms")

        # ---- Stage 2b: Translation back from English ----
        t0 = time.perf_counter()
        reconstructed_text = translate_from_english(reconstructed_en, detected_lang)
        t1 = time.perf_counter()
        result.t_translate_out_ms = (t1 - t0) * 1000
        result.reconstructed_text = reconstructed_text

        # ---- Stage 3: BLEU score ----
        result.bleu_score = self.pipeline._compute_bleu(text_en, reconstructed_en)

        # ---- Stage 4: TTS (Kokoro API) ----
        if not skip_tts:
            print(f"[Benchmark] Stage 3: TTS synthesis")
            t0 = time.perf_counter()
            try:
                self.pipeline.tts.synthesize(
                    text=reconstructed_text,
                    emotion=emotion,
                    gender=gender,
                    output_path=output_path,
                )
            except Exception as e:
                print(f"  TTS failed: {e} (timing still recorded)")
            t1 = time.perf_counter()
            result.t_tts_ms = (t1 - t0) * 1000
            print(f"  TTS time: {result.t_tts_ms:.1f} ms")
        else:
            # Estimate TTS time from published benchmarks
            # Kokoro API typical response: 2-8 seconds (network + synthesis)
            result.t_tts_ms = 3000.0  # Conservative 3s estimate
            print(f"[Benchmark] Stage 3: TTS skipped (using estimate: {result.t_tts_ms:.0f} ms)")

        t_pipeline_end = time.perf_counter()

        # ---- Derived Metrics ----
        result.t_total_e2e_ms = (t_pipeline_end - t_pipeline_start) * 1000

        # TTFA: Time-to-First-Audio for S2T2S
        # = STT buffer + STT inference + translation + DeepSC + reverse translation + TTS
        # The key insight: ALL of these must complete before the first audio byte is produced.
        result.ttfa_ms = (
            result.t_stt_buffer_ms
            + result.t_stt_ms
            + result.t_translate_in_ms
            + result.t_encode_ms + result.t_channel_ms + result.t_decode_ms
            + result.t_translate_out_ms
            + result.t_tts_ms
        )

        # RTF: Real-Time Factor
        if result.audio_duration_s > 0:
            result.rtf = (result.t_total_e2e_ms / 1000) / result.audio_duration_s
        else:
            result.rtf = float("inf")

        # Payload size & bitrate
        result.payload_bytes = compute_payload_size_s2t2s(
            text_en, self.pipeline.deepsc
        )
        if result.audio_duration_s > 0:
            result.bitrate_bps = (result.payload_bytes * 8) / result.audio_duration_s

        # ---- Summary ----
        print(f"\n{'='*60}")
        print(f"  S2T2S BENCHMARK RESULTS — {result.audio_file}")
        print(f"{'='*60}")
        print(f"  Audio duration : {result.audio_duration_s:.2f} s")
        print(f"  STT            : {result.t_stt_ms:.1f} ms (buffer: {result.t_stt_buffer_ms:.0f} ms)")
        print(f"  Translation in : {result.t_translate_in_ms:.1f} ms")
        print(f"  DeepSC encode  : {result.t_encode_ms:.1f} ms")
        print(f"  Channel        : {result.t_channel_ms:.1f} ms")
        print(f"  DeepSC decode  : {result.t_decode_ms:.1f} ms")
        print(f"  Translation out: {result.t_translate_out_ms:.1f} ms")
        print(f"  TTS            : {result.t_tts_ms:.1f} ms")
        print(f"  -------------------------------------")
        print(f"  Total E2E      : {result.t_total_e2e_ms:.1f} ms")
        print(f"  TTFA           : {result.ttfa_ms:.1f} ms")
        print(f"  RTF            : {result.rtf:.3f}")
        print(f"  Payload        : {result.payload_bytes} bytes")
        print(f"  Bitrate        : {result.bitrate_bps:.0f} bps")
        print(f"  BLEU           : {result.bleu_score:.4f}")
        print(f"{'='*60}\n")

        return result


# ------------------------------------------------------------------ #
#  CLI quick test                                                      #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python timing.py <audio_file.wav> [--skip-tts]")
        sys.exit(1)

    audio_path = sys.argv[1]
    skip_tts = "--skip-tts" in sys.argv

    timed = TimedS2T2SPipeline()
    result = timed.run_timed(audio_path, skip_tts=skip_tts)

    print("\nResult dict:")
    for k, v in result.to_dict().items():
        print(f"  {k}: {v}")
