"""
Bandwidth Compression Dashboard Module.

Compares the bandwidth usage of traditional communication systems
(source coding + channel coding) versus DeepSC semantic communication.

This answers the real-world question: "How much bandwidth does
semantic communication save compared to traditional methods?"
"""

import math


def compute_bandwidth_stats(
    text: str,
    deepsc_wrapper=None,
    assumed_bandwidth_bps: float = 1_000_000,  # 1 Mbps
) -> dict:
    """
    Calculate bandwidth comparison between traditional and semantic
    communication systems.

    Traditional System:
        1. Source coding: UTF-8 encoding (variable length)
        2. Compression: ~60% of raw (Huffman/arithmetic estimate)
        3. Channel coding: rate-1/2 LDPC/Turbo code (2× overhead)
        4. Total: UTF-8 bytes × 8 bits × 0.6 (compression) × 2 (channel coding)

    Semantic System (DeepSC):
        1. Tokenize: N words → N+2 tokens (with <START>, <END>)
        2. Semantic encode: each token → 16-dim channel symbol (float32)
        3. Total: (N+2) × 16 × 32 bits
        4. Note: DeepSC preserves *meaning*, not exact text

    Parameters
    ----------
    text : str
        Input text to analyze.
    deepsc_wrapper : DeepSCWrapper, optional
        If provided, uses actual model dimensions. Otherwise uses defaults.
    assumed_bandwidth_bps : float
        Assumed wireless link bandwidth in bits per second.

    Returns
    -------
    dict with keys:
        - text_length: int (character count)
        - word_count: int
        - traditional_raw_bits: int
        - traditional_compressed_bits: int
        - traditional_coded_bits: int (after channel coding)
        - semantic_tokens: int
        - semantic_symbols: int (total channel symbols)
        - semantic_bits: int (at float32 precision)
        - compression_ratio: float
        - savings_percent: float
        - traditional_time_us: float (microseconds at assumed bandwidth)
        - semantic_time_us: float
    """
    text_bytes = text.encode("utf-8")
    text_length = len(text)
    word_count = len(text.split())

    # --- Traditional System ---
    raw_bits = len(text_bytes) * 8
    # Estimate compression (Huffman/arithmetic typically achieves ~60% of raw)
    compressed_bits = int(raw_bits * 0.6)
    # Channel coding overhead (rate-1/2)
    coded_bits_traditional = compressed_bits * 2

    # --- Semantic System (DeepSC) ---
    # DeepSC channel encoder output: 16 dimensions per token
    channel_dim = 16
    if deepsc_wrapper and hasattr(deepsc_wrapper, "token_to_idx") and deepsc_wrapper.token_to_idx:
        # Tokenize using actual vocabulary
        from pipeline.semantic.deepsc_wrapper import _normalize_string, _tokenize
        normalized = _normalize_string(text)
        tokens = _tokenize(normalized)
        token_count = min(len(tokens), deepsc_wrapper.MAX_LEN)
    else:
        # Estimate: ~1.2 tokens per word + START/END
        token_count = int(word_count * 1.2) + 2

    total_symbols = token_count * channel_dim
    # Each symbol is a float32 (32 bits), but in practice could be quantized
    # We use float32 as upper bound and also show a quantized (8-bit) estimate
    semantic_bits_f32 = total_symbols * 32
    semantic_bits_q8 = total_symbols * 8  # If quantized to 8-bit

    # --- Compression Ratio ---
    # To compare fairly, we compare Channel Uses (Symbols)
    # Traditional: 16-QAM uses 4 bits per symbol
    traditional_channel_uses = coded_bits_traditional / 4
    semantic_channel_uses = total_symbols

    compression_ratio = traditional_channel_uses / semantic_channel_uses if semantic_channel_uses > 0 else 0
    savings_percent = max(0, (1 - semantic_channel_uses / traditional_channel_uses) * 100) if traditional_channel_uses > 0 else 0

    # --- Transmission Time ---
    # Assume 1 MHz symbol rate
    symbol_rate = 1_000_000
    traditional_time_us = (traditional_channel_uses / symbol_rate) * 1e6
    semantic_time_us = (semantic_channel_uses / symbol_rate) * 1e6

    return {
        "text_length": text_length,
        "word_count": word_count,
        # Traditional
        "traditional_raw_bits": raw_bits,
        "traditional_coded_bits": coded_bits_traditional,
        "traditional_channel_uses": traditional_channel_uses,
        # Semantic
        "semantic_tokens": token_count,
        "semantic_symbol_dim": channel_dim,
        "semantic_channel_uses": semantic_channel_uses,
        # Comparison
        "compression_ratio": round(compression_ratio, 2),
        "savings_percent": round(savings_percent, 1),
        # Timing
        "traditional_time_us": round(traditional_time_us, 1),
        "semantic_time_us": round(semantic_time_us, 1),
        "symbol_rate_mhz": symbol_rate / 1e6,
    }


def format_bandwidth_html(stats: dict) -> str:
    """
    Generate an HTML infographic for the bandwidth comparison.
    Uses CSS classes from custom_theme.css (bw-* classes).
    """
    trad_uses = stats["traditional_channel_uses"]
    sem_uses  = stats["semantic_channel_uses"]
    max_uses  = max(trad_uses, sem_uses, 1)

    trad_pct = round((trad_uses / max_uses) * 100, 1)
    sem_pct  = round((sem_uses  / max_uses) * 100, 1)

    savings = stats["savings_percent"]
    ratio   = stats["compression_ratio"]

    return f"""
    <div class="bw-dashboard">

        <div class="bw-stats">
            <div class="bw-stat">
                <div class="bw-stat-val">{stats['word_count']}</div>
                <div class="bw-stat-lbl">Words</div>
            </div>
            <div class="bw-stat">
                <div class="bw-stat-val">{stats['semantic_tokens']}</div>
                <div class="bw-stat-lbl">Semantic Tokens</div>
            </div>
            <div class="bw-stat highlight">
                <div class="bw-stat-val">{savings}%</div>
                <div class="bw-stat-lbl">Bandwidth Saved</div>
            </div>
            <div class="bw-stat">
                <div class="bw-stat-val">{ratio}x</div>
                <div class="bw-stat-lbl">Compression</div>
            </div>
        </div>

        <div class="bw-bars">
            <div class="bw-bar-row">
                <div class="bw-bar-meta">
                    <span class="bw-method">Traditional</span>
                    <span class="bw-detail">UTF-8 + Rate-1/2 LDPC + 16-QAM</span>
                </div>
                <div class="bw-track">
                    <div class="bw-bar-inner trad" style="width:{trad_pct}%">
                        <span class="bw-bar-lbl">{int(trad_uses):,} Channel Uses</span>
                    </div>
                </div>
            </div>
            <div class="bw-bar-row">
                <div class="bw-bar-meta">
                    <span class="bw-method">DeepSC Semantic</span>
                    <span class="bw-detail">{stats['semantic_tokens']} tokens x {stats['semantic_symbol_dim']} symbols</span>
                </div>
                <div class="bw-track">
                    <div class="bw-bar-inner semantic" style="width:{sem_pct}%">
                        <span class="bw-bar-lbl">{int(sem_uses):,} Channel Uses</span>
                    </div>
                </div>
            </div>
        </div>

        <div class="bw-timing">
            <div class="bw-time">
                <div class="bw-time-lbl">Traditional Tx</div>
                <div class="bw-time-val">{stats['traditional_time_us']} &mu;s</div>
            </div>
            <div class="bw-time">
                <div class="bw-time-lbl">Semantic Tx</div>
                <div class="bw-time-val">{stats['semantic_time_us']} &mu;s</div>
            </div>
            <div class="bw-time">
                <div class="bw-time-lbl">Symbol Rate</div>
                <div class="bw-time-val">{stats['symbol_rate_mhz']} Msps</div>
            </div>
        </div>

    </div>
    """
