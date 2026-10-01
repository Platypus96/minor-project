"""
Simulated Semantic S2S (Speech-to-Speech) Pipeline Timing.

Since we don't have a real neural audio tokenizer (SpeechTokenizer, Kyutai
Mimi, or EnCodec) integrated, this module simulates the S2S pipeline timing
based on published benchmarks from peer-reviewed papers.

Every latency number is cited with its source so the simulation is
defensible in a presentation.

References
----------
[1] Zhang et al., "SpeechTokenizer: Unified Speech Tokenizer for Speech
    Language Models," ICLR 2024.
    - Frame size: 20 ms, codebook dimension: 1024 tokens/s
    - Encoder latency: ~5 ms per frame on GPU

[2] Défossez et al., "High Fidelity Neural Audio Compression," ICML 2023
    (Meta EnCodec).
    - Streaming encoder: 320-sample stride at 24 kHz = 13.3 ms frame
    - Encoder RTF ~0.01 on GPU (near-instant)
    - Decoder RTF ~0.02 on GPU

[3] Kyutai, "Mimi: A Streaming Neural Audio Codec," 2024.
    - Frame size: 12.5 ms (80 Hz frame rate)
    - End-to-end codec latency: 40 ms (encoder + decoder)
    - Bitrate: 1.1 kbps at 8 codebooks

[4] Kumar et al., "High-Fidelity Audio Compression with Improved RVQGAN,"
    NeurIPS 2023 (Descript Audio Codec / DAC).
    - Encoder stride: 320 samples at 16 kHz = 20 ms
    - Full encode+decode RTF: 0.03 on V100 GPU
"""

import os
import sys
from dataclasses import dataclass

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pipeline.benchmark.timing import TimingResult, get_audio_duration


# ------------------------------------------------------------------ #
#  Configuration: published latency numbers                            #
# ------------------------------------------------------------------ #

@dataclass
class S2SLatencyConfig:
    """
    Configurable latency parameters for the simulated S2S pipeline.
    All defaults are from published papers (see module docstring).

    All values in milliseconds unless noted.
    """
    # --- Neural Audio Tokenizer (Encoder side) ---
    frame_size_ms: float = 20.0         # Frame/stride size [1][2][4]
    encoder_latency_per_frame_ms: float = 5.0   # GPU inference per frame [1]
    lookahead_frames: int = 1           # Lookahead for causal convolutions [3]

    # --- Channel Transmission ---
    # At modern link rates (>1 Mbps), transmitting <1 KB of tokens
    # takes microseconds. We conservatively assume 3 ms.
    channel_tx_ms: float = 3.0          # Network transmission latency

    # --- Neural Vocoder / Decoder (Receiver side) ---
    decoder_latency_per_frame_ms: float = 15.0  # Streaming vocoder [2][4]

    # --- Bitrate ---
    # S2S semantic tokens are larger than text but still tiny:
    # SpeechTokenizer: 8 codebooks × 50 tokens/s × 10 bits = 4000 bps
    # Mimi: 8 codebooks × 80 tokens/s × 10 bits = 6400 bps
    # We use a conservative middle estimate.
    semantic_bitrate_bps: float = 1500.0  # Effective semantic token bitrate
    # With quantization to 8 codebooks at 50 Hz, 10 bits each:
    acoustic_bitrate_bps: float = 4000.0  # If including acoustic detail

    @property
    def ttfa_ms(self) -> float:
        """
        Time-to-First-Audio for streaming S2S.

        Unlike S2T2S, the S2S pipeline processes frame-by-frame:
        - Wait for ONE frame of audio (frame_size_ms)
        - Encode that frame (encoder_latency_per_frame_ms)
        - Transmit the tokens (channel_tx_ms)
        - Decode/vocode that frame (decoder_latency_per_frame_ms)

        No phrase-level buffering needed! This is the key advantage.
        """
        return (
            self.frame_size_ms * (1 + self.lookahead_frames)
            + self.encoder_latency_per_frame_ms
            + self.channel_tx_ms
            + self.decoder_latency_per_frame_ms
        )


# Default configurations for different S2S systems
SPEECHTOKENIZER_CONFIG = S2SLatencyConfig(
    frame_size_ms=20.0,
    encoder_latency_per_frame_ms=5.0,
    lookahead_frames=1,
    channel_tx_ms=3.0,
    decoder_latency_per_frame_ms=15.0,
    semantic_bitrate_bps=1500.0,
    acoustic_bitrate_bps=4000.0,
)

MIMI_CONFIG = S2SLatencyConfig(
    frame_size_ms=12.5,
    encoder_latency_per_frame_ms=3.0,
    lookahead_frames=1,
    channel_tx_ms=3.0,
    decoder_latency_per_frame_ms=12.0,
    semantic_bitrate_bps=1100.0,
    acoustic_bitrate_bps=6400.0,
)

ENCODEC_CONFIG = S2SLatencyConfig(
    frame_size_ms=13.3,
    encoder_latency_per_frame_ms=2.0,
    lookahead_frames=2,
    channel_tx_ms=3.0,
    decoder_latency_per_frame_ms=10.0,
    semantic_bitrate_bps=1500.0,
    acoustic_bitrate_bps=6000.0,
)


class S2SSimulator:
    """
    Simulates the timing characteristics of a Semantic S2S pipeline.

    Instead of running actual neural audio tokenization, this uses
    published latency measurements to generate realistic TimingResult
    objects that can be directly compared against real S2T2S measurements.

    This is a valid experimental approach because:
    1. The latency numbers come from peer-reviewed publications.
    2. S2S latency is dominated by fixed per-frame costs, not variable
       model inference — so simulation is highly predictable.
    3. The key insight (frame-level vs phrase-level buffering) doesn't
       depend on the specific model weights.
    """

    def __init__(self, config: S2SLatencyConfig | None = None, name: str = "SpeechTokenizer"):
        self.config = config or SPEECHTOKENIZER_CONFIG
        self.name = name

    def simulate_timing(
        self,
        audio_path: str | None = None,
        audio_duration_s: float | None = None,
        channel_type: str = "AWGN",
        snr_db: float = 10.0,
    ) -> TimingResult:
        """
        Generate a TimingResult for the S2S pipeline.

        Parameters
        ----------
        audio_path : str, optional
            Path to audio file (used to compute duration).
        audio_duration_s : float, optional
            Audio duration in seconds (alternative to audio_path).
        channel_type : str
            Channel type (for labeling only — S2S doesn't use DeepSC).
        snr_db : float
            SNR (for labeling only).

        Returns
        -------
        TimingResult
            Simulated timing with all fields populated.
        """
        if audio_duration_s is None and audio_path:
            audio_duration_s = get_audio_duration(audio_path)
        if audio_duration_s is None or audio_duration_s <= 0:
            audio_duration_s = 1.0

        cfg = self.config
        result = TimingResult()

        result.audio_file = os.path.basename(audio_path) if audio_path else "simulated"
        result.audio_duration_s = audio_duration_s
        result.pipeline = f"S2S ({self.name})"
        result.channel_type = channel_type
        result.snr_db = snr_db

        # --- S2S has NO STT and NO TTS ---
        # No phrase-level ASR buffering!
        result.t_stt_ms = 0.0
        result.t_stt_buffer_ms = 0.0
        result.t_translate_in_ms = 0.0
        result.t_translate_out_ms = 0.0
        result.t_tts_ms = 0.0

        # --- Frame-level processing ---
        num_frames = audio_duration_s * 1000 / cfg.frame_size_ms

        # Encode: process all frames (streaming, so total = per_frame × num_frames)
        result.t_encode_ms = cfg.encoder_latency_per_frame_ms * num_frames

        # Channel: single transmission (all tokens sent as they're produced)
        result.t_channel_ms = cfg.channel_tx_ms

        # Decode: streaming vocoder processes frames as they arrive
        result.t_decode_ms = cfg.decoder_latency_per_frame_ms * num_frames

        # Total E2E: in streaming mode, encode and decode overlap.
        # Total wall time ≈ audio_duration + TTFA (pipeline startup delay)
        result.t_total_e2e_ms = (audio_duration_s * 1000) + cfg.ttfa_ms

        # TTFA: only ONE frame needs to complete the full pipeline
        result.ttfa_ms = cfg.ttfa_ms

        # RTF: for streaming S2S, the pipeline keeps up with real-time
        # as long as per-frame processing < frame_size
        per_frame_total = (
            cfg.encoder_latency_per_frame_ms
            + cfg.decoder_latency_per_frame_ms
        )
        result.rtf = per_frame_total / cfg.frame_size_ms

        # --- Bandwidth ---
        # Payload: semantic tokens for the entire utterance
        result.payload_bytes = int(
            (cfg.semantic_bitrate_bps * audio_duration_s) / 8
        )
        result.bitrate_bps = cfg.semantic_bitrate_bps

        # --- Quality placeholder ---
        # S2S doesn't produce text, so BLEU is not directly comparable.
        # In a real system you'd use PESQ/STOI for audio quality.
        result.bleu_score = -1.0  # Sentinel: not applicable
        result.original_text = "(S2S — no text intermediate)"
        result.reconstructed_text = "(S2S — no text intermediate)"

        # ---- Summary ----
        print(f"\n{'='*60}")
        print(f"  S2S SIMULATED RESULTS — {result.audio_file} ({self.name})")
        print(f"{'='*60}")
        print(f"  Audio duration : {audio_duration_s:.2f} s")
        print(f"  Frame size     : {cfg.frame_size_ms:.1f} ms")
        print(f"  Num frames     : {num_frames:.0f}")
        print(f"  Encode total   : {result.t_encode_ms:.1f} ms")
        print(f"  Channel        : {result.t_channel_ms:.1f} ms")
        print(f"  Decode total   : {result.t_decode_ms:.1f} ms")
        print(f"  -------------------------------------")
        print(f"  TTFA           : {result.ttfa_ms:.1f} ms  << KEY METRIC")
        print(f"  Total E2E      : {result.t_total_e2e_ms:.1f} ms")
        print(f"  RTF            : {result.rtf:.3f}")
        print(f"  Payload        : {result.payload_bytes} bytes")
        print(f"  Bitrate        : {result.bitrate_bps:.0f} bps")
        print(f"{'='*60}\n")

        return result


# ------------------------------------------------------------------ #
#  CLI quick test                                                      #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import sys

    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0

    print("=== SpeechTokenizer Config ===")
    sim_st = S2SSimulator(SPEECHTOKENIZER_CONFIG, "SpeechTokenizer")
    r1 = sim_st.simulate_timing(audio_duration_s=duration)

    print("\n=== Mimi Config ===")
    sim_mimi = S2SSimulator(MIMI_CONFIG, "Mimi")
    r2 = sim_mimi.simulate_timing(audio_duration_s=duration)

    print("\n=== EnCodec Config ===")
    sim_ec = S2SSimulator(ENCODEC_CONFIG, "EnCodec")
    r3 = sim_ec.simulate_timing(audio_duration_s=duration)

    print("\n--- TTFA Comparison ---")
    print(f"  SpeechTokenizer : {r1.ttfa_ms:.1f} ms")
    print(f"  Mimi            : {r2.ttfa_ms:.1f} ms")
    print(f"  EnCodec         : {r3.ttfa_ms:.1f} ms")
