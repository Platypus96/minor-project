"""
SNR Sweep Performance Analysis Module.

Generates BLEU-vs-SNR performance curves — the classic evaluation
metric used in every semantic communication research paper.

This answers the real-world question: "How robust is our semantic
communication system under varying noise conditions?"
"""

import os
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Use non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _compute_bleu(reference: str, hypothesis: str) -> float:
    """Compute BLEU score using DeepSC's normalization."""
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


def snr_sweep(
    text: str,
    deepsc_wrapper,
    channel_types: list[str] | None = None,
    snr_min: float = -5.0,
    snr_max: float = 20.0,
    snr_step: float = 2.5,
    progress_callback=None,
) -> dict:
    """
    Run DeepSC at multiple SNR points and compute BLEU at each.

    Parameters
    ----------
    text : str
        Input text (English) to transmit.
    deepsc_wrapper : DeepSCWrapper
        The loaded wrapper instance.
    channel_types : list[str], optional
        Channels to sweep. Default: all three.
    snr_min, snr_max, snr_step : float
        SNR range parameters.
    progress_callback : callable, optional
        Called with (current_step, total_steps) for UI progress updates.

    Returns
    -------
    dict
        {
            "snr_values": [float, ...],
            "channels": {
                "AWGN": [bleu_at_snr0, bleu_at_snr1, ...],
                "RAYLEIGH": [...],
                "RICIAN": [...],
            }
        }
    """
    if channel_types is None:
        channel_types = ["AWGN", "RAYLEIGH", "RICIAN"]

    snr_values = list(np.arange(snr_min, snr_max + snr_step / 2, snr_step))
    total_steps = len(snr_values) * len(channel_types)
    current_step = 0

    # Save original config
    original_channel = deepsc_wrapper.channel.channel_type
    original_snr = deepsc_wrapper.channel.snr_db

    results = {"snr_values": snr_values, "channels": {}}

    for ch in channel_types:
        ch_upper = ch.upper()
        bleu_scores = []

        for snr in snr_values:
            try:
                deepsc_wrapper.set_channel(ch_upper, snr)
                reconstructed = deepsc_wrapper.transmit(text)
                bleu = _compute_bleu(text, reconstructed)
                bleu_scores.append(bleu)
            except Exception as e:
                print(f"[SNR Sweep] Error at {ch_upper}/{snr}dB: {e}")
                bleu_scores.append(0.0)

            current_step += 1
            if progress_callback:
                progress_callback(current_step, total_steps)

        results["channels"][ch_upper] = bleu_scores

    # Restore original config
    deepsc_wrapper.set_channel(original_channel, original_snr)

    return results


def plot_snr_curve(
    sweep_results: dict,
    output_path: str = "snr_curve.png",
) -> str:
    """
    Generate a publication-quality BLEU-vs-SNR performance curve.

    Parameters
    ----------
    sweep_results : dict
        Output from snr_sweep().
    output_path : str
        Where to save the PNG.

    Returns
    -------
    str
        Path to the saved image.
    """
    snr_values = sweep_results["snr_values"]
    channels = sweep_results["channels"]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")

    # Channel styles
    styles = {
        "AWGN": {"color": "#4361ee", "marker": "o", "label": "AWGN"},
        "RAYLEIGH": {"color": "#7c3aed", "marker": "s", "label": "Rayleigh Fading"},
        "RICIAN": {"color": "#0891b2", "marker": "^", "label": "Rician Fading (K=1)"},
    }

    for ch_name, bleu_scores in channels.items():
        style = styles.get(ch_name.upper(), {"color": "#64748b", "marker": "x", "label": ch_name})
        ax.plot(
            snr_values, bleu_scores,
            color=style["color"],
            marker=style["marker"],
            markersize=7,
            linewidth=2,
            label=style["label"],
            alpha=0.9,
        )

    ax.set_xlabel("SNR (dB)", fontsize=12, fontweight="500", color="#1e293b", labelpad=10)
    ax.set_ylabel("BLEU Score", fontsize=12, fontweight="500", color="#1e293b", labelpad=10)
    ax.set_title(
        "Semantic Fidelity vs. Channel Quality",
        fontsize=15, fontweight="700", color="#1e293b", pad=15,
    )

    ax.legend(
        frameon=True, fancybox=True, framealpha=0.9,
        edgecolor="#e2e8f0", fontsize=10,
        loc="lower right",
    )

    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, color="#94a3b8")
    ax.tick_params(colors="#64748b", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#e2e8f0")

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    return output_path
