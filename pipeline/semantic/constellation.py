"""
Signal Constellation Diagram Generator.

Generates I/Q scatter plots of transmitted symbols before and after
passing through a wireless channel. This is a standard visualization
in wireless communications that shows how channel noise distorts the
signal constellation.
"""

import os
import numpy as np

# Use non-interactive backend so matplotlib doesn't need a display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def plot_constellation(
    pre_symbols: np.ndarray,
    post_symbols: np.ndarray,
    channel_type: str = "AWGN",
    snr_db: float = 10.0,
    output_path: str = "constellation.png",
) -> str:
    """
    Generate a constellation diagram showing symbols before and after
    channel corruption.

    Parameters
    ----------
    pre_symbols : np.ndarray
        Transmitted symbols (flattened to pairs of I/Q components).
    post_symbols : np.ndarray
        Received symbols after channel effects.
    channel_type : str
        Channel type label for the title.
    snr_db : float
        SNR value for the title.
    output_path : str
        Where to save the PNG.

    Returns
    -------
    str
        Path to the saved image.
    """
    # Reshape to I/Q pairs: treat consecutive values as (I, Q)
    pre = pre_symbols.flatten()
    post = post_symbols.flatten()

    # Ensure even length
    if len(pre) % 2 != 0:
        pre = pre[:-1]
    if len(post) % 2 != 0:
        post = post[:-1]

    pre_i = pre[0::2]
    pre_q = pre[1::2]
    post_i = post[0::2]
    post_q = post[1::2]

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=120)
    fig.patch.set_facecolor("#f8fafc")

    # Color palette (light theme friendly)
    tx_color = "#4361ee"
    rx_color = "#ef4444"

    # Transmitted symbols
    ax1 = axes[0]
    ax1.scatter(pre_i, pre_q, s=12, alpha=0.7, c=tx_color, edgecolors="none")
    ax1.set_title("Transmitted Symbols", fontsize=13, fontweight="600", color="#1e293b")
    ax1.set_xlabel("In-Phase (I)", fontsize=10, color="#64748b")
    ax1.set_ylabel("Quadrature (Q)", fontsize=10, color="#64748b")
    ax1.axhline(y=0, color="#cbd5e1", linewidth=0.5)
    ax1.axvline(x=0, color="#cbd5e1", linewidth=0.5)
    ax1.set_facecolor("#f8fafc")
    ax1.grid(True, alpha=0.3, color="#94a3b8")
    ax1.tick_params(colors="#64748b", labelsize=9)
    for spine in ax1.spines.values():
        spine.set_color("#e2e8f0")

    # Received symbols
    ax2 = axes[1]
    ax2.scatter(post_i, post_q, s=12, alpha=0.7, c=rx_color, edgecolors="none")
    ax2.set_title("Received Symbols (After Channel)", fontsize=13, fontweight="600", color="#1e293b")
    ax2.set_xlabel("In-Phase (I)", fontsize=10, color="#64748b")
    ax2.set_ylabel("Quadrature (Q)", fontsize=10, color="#64748b")
    ax2.axhline(y=0, color="#cbd5e1", linewidth=0.5)
    ax2.axvline(x=0, color="#cbd5e1", linewidth=0.5)
    ax2.set_facecolor("#f8fafc")
    ax2.grid(True, alpha=0.3, color="#94a3b8")
    ax2.tick_params(colors="#64748b", labelsize=9)
    for spine in ax2.spines.values():
        spine.set_color("#e2e8f0")

    fig.suptitle(
        f"Signal Constellation — {channel_type} Channel @ {snr_db} dB SNR",
        fontsize=14,
        fontweight="700",
        color="#1e293b",
        y=1.02,
    )
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    return output_path


def plot_constellation_comparison(
    results: dict,
    snr_db: float = 10.0,
    output_path: str = "constellation_comparison.png",
) -> str:
    """
    Generate a 3-column constellation comparison (one per channel).

    Parameters
    ----------
    results : dict
        {channel_name: {"pre_symbols": np.ndarray, "post_symbols": np.ndarray}}
    snr_db : float
        SNR value for the title.
    output_path : str
        Where to save the PNG.

    Returns
    -------
    str
        Path to the saved image.
    """
    channels = list(results.keys())
    n = len(channels)
    if n == 0:
        return ""

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), dpi=120)
    fig.patch.set_facecolor("#f8fafc")

    if n == 1:
        axes = [axes]

    colors = {"AWGN": "#4361ee", "RAYLEIGH": "#7c3aed", "RICIAN": "#0891b2"}

    for idx, ch_name in enumerate(channels):
        ax = axes[idx]
        data = results[ch_name]
        post = data["post_symbols"].flatten()
        if len(post) % 2 != 0:
            post = post[:-1]
        post_i = post[0::2]
        post_q = post[1::2]

        color = colors.get(ch_name.upper(), "#64748b")
        ax.scatter(post_i, post_q, s=14, alpha=0.6, c=color, edgecolors="none")
        ax.set_title(f"{ch_name}", fontsize=13, fontweight="600", color="#1e293b")
        ax.set_xlabel("I", fontsize=10, color="#64748b")
        ax.set_ylabel("Q", fontsize=10, color="#64748b")
        ax.axhline(y=0, color="#cbd5e1", linewidth=0.5)
        ax.axvline(x=0, color="#cbd5e1", linewidth=0.5)
        ax.set_facecolor("#f8fafc")
        ax.grid(True, alpha=0.3, color="#94a3b8")
        ax.tick_params(colors="#64748b", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#e2e8f0")

    fig.suptitle(
        f"Constellation Comparison @ {snr_db} dB SNR",
        fontsize=14, fontweight="700", color="#1e293b", y=1.02,
    )
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    return output_path
