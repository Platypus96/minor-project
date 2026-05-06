"""
Multi-Channel Comparison Module.

Runs the same input text through all available wireless channels
(AWGN, Rayleigh, Rician) at the same SNR and returns comparative
results — BLEU scores, reconstructed text, and symbol data for
constellation diagrams.

This answers the real-world question: "Which channel environment
performs best for semantic communication?"
"""

import os
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def _compute_bleu(reference: str, hypothesis: str) -> float:
    """Compute BLEU score between two strings using DeepSC's normalization."""
    try:
        from pipeline.semantic.deepsc_wrapper import _normalize_string
        smoothie = SmoothingFunction().method1
        ref_tokens = _normalize_string(reference).split()
        hyp_tokens = _normalize_string(hypothesis).split()
        if not ref_tokens or not hyp_tokens:
            return 0.0
        return sentence_bleu(
            [ref_tokens], hyp_tokens, smoothing_function=smoothie,
        )
    except Exception:
        return 0.0


def compare_channels(
    text: str,
    snr_db: float,
    deepsc_wrapper,
    channels: list[str] | None = None,
) -> dict:
    """
    Run the same text through multiple wireless channels and compare results.

    Parameters
    ----------
    text : str
        Input text to transmit (should be English for DeepSC).
    snr_db : float
        Signal-to-Noise Ratio in dB.
    deepsc_wrapper : DeepSCWrapper
        The loaded DeepSC wrapper instance.
    channels : list[str], optional
        Channel types to compare. Defaults to all three.

    Returns
    -------
    dict
        {
            "AWGN": {
                "reconstructed_text": str,
                "bleu_score": float,
                "pre_symbols": np.ndarray,
                "post_symbols": np.ndarray,
            },
            "RAYLEIGH": { ... },
            "RICIAN": { ... },
        }
    """
    if channels is None:
        channels = ["AWGN", "RAYLEIGH", "RICIAN"]

    # Save original channel config to restore later
    original_channel = deepsc_wrapper.channel.channel_type
    original_snr = deepsc_wrapper.channel.snr_db

    results = {}
    for ch in channels:
        ch_upper = ch.upper()
        try:
            # Switch channel
            deepsc_wrapper.set_channel(ch_upper, snr_db)

            # Transmit
            reconstructed = deepsc_wrapper.transmit(text)

            # Get symbols if available
            pre_sym, post_sym = deepsc_wrapper.get_last_symbols()

            # BLEU
            bleu = _compute_bleu(text, reconstructed)

            results[ch_upper] = {
                "reconstructed_text": reconstructed,
                "bleu_score": bleu,
                "pre_symbols": pre_sym if pre_sym is not None else np.array([]),
                "post_symbols": post_sym if post_sym is not None else np.array([]),
            }
        except Exception as e:
            print(f"[Comparison] Error on {ch_upper}: {e}")
            results[ch_upper] = {
                "reconstructed_text": f"Error: {e}",
                "bleu_score": 0.0,
                "pre_symbols": np.array([]),
                "post_symbols": np.array([]),
            }

    # Restore original channel config
    deepsc_wrapper.set_channel(original_channel, original_snr)

    return results
