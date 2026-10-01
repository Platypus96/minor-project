"""
Comprehensive Benchmark Runner: Bandwidth vs. Time Efficiency
Semantic S2S vs. S2T2S Pipelines

Executes the complete 2x2 benchmark matrix:
- Pipeline A: Semantic S2S (SpeechTokenizer, Mimi, EnCodec simulations)
- Pipeline B: Cascade S2T2S (Real STT + DeepSC Transformer + TTS)

Collects:
- Latency: TTFA (Time-to-First-Audio), algorithmic lookahead buffer, stage timings, RTF
- Bandwidth: Payload size (bytes), effective bitrate (bps)
- Fidelity: BLEU score across SNR and channel fading regimes

Outputs:
- outputs/benchmark_results.csv: Raw per-sample, per-channel, per-SNR results
- outputs/benchmark_summary.json: Aggregated statistics and comparative ratios
"""

import os
import sys
import json
import csv
import argparse
from typing import List, Dict, Any

# Ensure UTF-8 console output on Windows
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Add project root to path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pipeline.benchmark.timing import TimedS2T2SPipeline, TimingResult, get_audio_duration
from pipeline.benchmark.s2s_simulator import (
    S2SSimulator,
    SPEECHTOKENIZER_CONFIG,
    MIMI_CONFIG,
    ENCODEC_CONFIG,
)
from pipeline.benchmark.corpus import TestCorpus, AudioSample


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Bandwidth vs. Time Efficiency Benchmark (S2S vs S2T2S)"
    )
    parser.add_argument(
        "--corpus-dir",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "test_audio"),
        help="Directory containing test audio corpus",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "outputs", "benchmark_results.csv"),
        help="Path for saving CSV benchmark results",
    )
    parser.add_argument(
        "--output-summary",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "outputs", "benchmark_summary.json"),
        help="Path for saving JSON summary statistics",
    )
    parser.add_argument(
        "--channels",
        type=str,
        default="AWGN,RAYLEIGH,RICIAN",
        help="Comma-separated wireless channels (AWGN, RAYLEIGH, RICIAN)",
    )
    parser.add_argument(
        "--snr-values",
        type=str,
        default="10.0",
        help="Comma-separated SNR values in dB for latency benchmark (e.g. 10.0 or -5,0,5,10,15,20)",
    )
    parser.add_argument(
        "--snr-sweep",
        action="store_true",
        help="Run full SNR sweep (-5 to 20 dB in steps of 5 dB) across channels",
    )
    parser.add_argument(
        "--skip-tts",
        action="store_true",
        default=True,
        help="Skip actual Kokoro TTS API requests and use estimated synthesis time (default: True)",
    )
    parser.add_argument(
        "--with-tts",
        action="store_false",
        dest="skip_tts",
        help="Execute real Kokoro TTS API calls (requires active TTS_API_KEY and network)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick run on 3 representative duration samples (short, medium, long) on AWGN 10dB",
    )
    return parser.parse_args()


def get_samples_to_run(corpus: TestCorpus, quick: bool = False) -> List[AudioSample]:
    """Select corpus samples for benchmarking."""
    available = corpus.get_available()
    if not available:
        raise RuntimeError("No available audio samples found in corpus!")

    if quick:
        # Pick one short, one medium, one long
        selected = []
        for cat in ["short", "medium", "long"]:
            cat_samples = [s for s in available if s.category == cat and s.language == "en"]
            if not cat_samples:
                cat_samples = [s for s in available if s.category == cat]
            if cat_samples:
                selected.append(cat_samples[0])
        return selected

    # Return all available English samples plus non-English
    return available


def run_benchmark(args):
    """Execute the full benchmark suite."""
    print("=" * 70)
    print("  BANDWIDTH VS. TIME EFFICIENCY BENCHMARK")
    print("  Semantic S2S vs. S2T2S Communication Pipelines")
    print("=" * 70)
    print(f"  Corpus directory : {args.corpus_dir}")
    print(f"  Output CSV       : {args.output_csv}")
    print(f"  Output Summary   : {args.output_summary}")
    print(f"  Skip TTS API     : {args.skip_tts}")

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_summary), exist_ok=True)

    # 1. Load Corpus
    corpus = TestCorpus(args.corpus_dir)
    samples = get_samples_to_run(corpus, quick=args.quick)
    print(f"\n[Corpus] Evaluating {len(samples)} audio samples:")
    for s in samples:
        print(f"  - {s.filename:<30} ({s.category:<6}, {s.duration_s:.2f}s, {s.language})")

    # 2. Parse channels and SNRs
    channels = [ch.strip().upper() for ch in args.channels.split(",") if ch.strip()]
    if args.snr_sweep:
        snr_values = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
    else:
        snr_values = [float(snr.strip()) for snr in args.snr_values.split(",") if snr.strip()]

    print(f"[Config] Channels: {channels}")
    print(f"[Config] SNR values (dB): {snr_values}")

    # 3. Initialize S2T2S Timed Pipeline (loads once into memory)
    print("\n[Init] Initializing S2T2S pipeline models (SenseVoice + DeepSC)...")
    s2t2s = TimedS2T2SPipeline()

    # 4. Initialize S2S Simulators
    simulators = {
        "SpeechTokenizer": S2SSimulator(SPEECHTOKENIZER_CONFIG, "SpeechTokenizer"),
        "Mimi": S2SSimulator(MIMI_CONFIG, "Mimi"),
        "EnCodec": S2SSimulator(ENCODEC_CONFIG, "EnCodec"),
    }

    # Storage for all run rows
    all_results: List[Dict[str, Any]] = []

    # 5. Run Benchmark Loops
    total_evals = len(samples) * len(channels) * len(snr_values)
    current_eval = 0

    print(f"\n[Benchmark] Beginning execution of {total_evals} test runs...")

    for sample in samples:
        # Determine language for pipeline
        lang_arg = sample.language if sample.language != "auto" else "auto"

        for channel in channels:
            for snr in snr_values:
                current_eval += 1
                print(f"\n>>> [{current_eval}/{total_evals}] Sample: {sample.filename} | Channel: {channel} | SNR: {snr} dB")

                # --- 5A. Pipeline B: S2T2S Execution ---
                try:
                    res_s2t2s = s2t2s.run_timed(
                        audio_input_path=sample.path,
                        channel=channel,
                        snr_db=snr,
                        language=lang_arg,
                        skip_tts=args.skip_tts,
                    )

                    row_s2t2s = {
                        "sample_name": sample.filename,
                        "category": sample.category,
                        "audio_duration_s": round(sample.duration_s, 3),
                        "language": sample.language,
                        "pipeline": "S2T2S",
                        "channel": channel,
                        "snr_db": snr,
                        "t_stt_ms": round(res_s2t2s.t_stt_ms, 2),
                        "t_stt_buffer_ms": round(res_s2t2s.t_stt_buffer_ms, 2),
                        "t_translate_in_ms": round(res_s2t2s.t_translate_in_ms, 2),
                        "t_encode_ms": round(res_s2t2s.t_encode_ms, 2),
                        "t_channel_ms": round(res_s2t2s.t_channel_ms, 2),
                        "t_decode_ms": round(res_s2t2s.t_decode_ms, 2),
                        "t_translate_out_ms": round(res_s2t2s.t_translate_out_ms, 2),
                        "t_tts_ms": round(res_s2t2s.t_tts_ms, 2),
                        "t_total_e2e_ms": round(res_s2t2s.t_total_e2e_ms, 2),
                        "ttfa_ms": round(res_s2t2s.ttfa_ms, 2),
                        "rtf": round(res_s2t2s.rtf, 4),
                        "payload_bytes": res_s2t2s.payload_bytes,
                        "bitrate_bps": round(res_s2t2s.bitrate_bps, 1),
                        "bleu_score": round(res_s2t2s.bleu_score, 4),
                        "original_text": res_s2t2s.original_text,
                        "reconstructed_text": res_s2t2s.reconstructed_text,
                    }
                    all_results.append(row_s2t2s)
                except Exception as e:
                    print(f"  [ERROR] S2T2S failed on {sample.filename} ({channel}, {snr}dB): {e}")

                # --- 5B. Pipeline A: Semantic S2S (Simulated) ---
                # S2S latency is channel/SNR independent for initial frame emission
                for sim_name, sim in simulators.items():
                    res_s2s = sim.simulate_timing(
                        audio_duration_s=sample.duration_s,
                        audio_path=sample.path,
                        channel_type=channel,
                        snr_db=snr,
                    )

                    row_s2s = {
                        "sample_name": sample.filename,
                        "category": sample.category,
                        "audio_duration_s": round(sample.duration_s, 3),
                        "language": sample.language,
                        "pipeline": f"S2S ({sim_name})",
                        "channel": channel,
                        "snr_db": snr,
                        "t_stt_ms": 0.0,
                        "t_stt_buffer_ms": 0.0,
                        "t_translate_in_ms": 0.0,
                        "t_encode_ms": round(res_s2s.t_encode_ms, 2),
                        "t_channel_ms": round(res_s2s.t_channel_ms, 2),
                        "t_decode_ms": round(res_s2s.t_decode_ms, 2),
                        "t_translate_out_ms": 0.0,
                        "t_tts_ms": 0.0,
                        "t_total_e2e_ms": round(res_s2s.t_total_e2e_ms, 2),
                        "ttfa_ms": round(res_s2s.ttfa_ms, 2),
                        "rtf": round(res_s2s.rtf, 4),
                        "payload_bytes": res_s2s.payload_bytes,
                        "bitrate_bps": round(res_s2s.bitrate_bps, 1),
                        "bleu_score": -1.0,
                        "original_text": "(S2S acoustic/semantic stream)",
                        "reconstructed_text": "(S2S neural vocoded audio)",
                    }
                    all_results.append(row_s2s)

    # 6. Save Raw CSV
    fieldnames = list(all_results[0].keys())
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n[Saved] CSV results written to: {args.output_csv}")

    # 7. Generate Summary Statistics
    summary = compute_summary(all_results)
    with open(args.output_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[Saved] Summary statistics written to: {args.output_summary}")

    # 8. Print Executive Summary Table
    print_executive_summary(summary)

    return all_results, summary


def compute_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate statistical comparisons."""
    pipelines = list(dict.fromkeys(r["pipeline"] for r in results))

    summary = {
        "total_records": len(results),
        "pipelines": {},
        "comparison": {},
        "categories": {},
        "snr_fidelity": {},
    }

    # Aggregate by pipeline
    for p in pipelines:
        subset = [r for r in results if r["pipeline"] == p]
        count = len(subset)
        if count == 0:
            continue

        mean_ttfa = sum(r["ttfa_ms"] for r in subset) / count
        mean_bitrate = sum(r["bitrate_bps"] for r in subset) / count
        mean_rtf = sum(r["rtf"] for r in subset) / count
        mean_payload = sum(r["payload_bytes"] for r in subset) / count

        summary["pipelines"][p] = {
            "count": count,
            "mean_ttfa_ms": round(mean_ttfa, 2),
            "min_ttfa_ms": round(min(r["ttfa_ms"] for r in subset), 2),
            "max_ttfa_ms": round(max(r["ttfa_ms"] for r in subset), 2),
            "mean_bitrate_bps": round(mean_bitrate, 1),
            "min_bitrate_bps": round(min(r["bitrate_bps"] for r in subset), 1),
            "max_bitrate_bps": round(max(r["bitrate_bps"] for r in subset), 1),
            "mean_rtf": round(mean_rtf, 4),
            "mean_payload_bytes": round(mean_payload, 1),
        }

        # BLEU for S2T2S
        bleu_vals = [r["bleu_score"] for r in subset if r["bleu_score"] >= 0]
        if bleu_vals:
            summary["pipelines"][p]["mean_bleu"] = round(sum(bleu_vals) / len(bleu_vals), 4)

    # Compute key comparative trade-off ratios
    s2t2s_stats = summary["pipelines"].get("S2T2S")
    s2s_st_stats = summary["pipelines"].get("S2S (SpeechTokenizer)")

    if s2t2s_stats and s2s_st_stats:
        ttfa_speedup = s2t2s_stats["mean_ttfa_ms"] / s2s_st_stats["mean_ttfa_ms"]
        bandwidth_ratio = s2s_st_stats["mean_bitrate_bps"] / s2t2s_stats["mean_bitrate_bps"]

        summary["comparison"] = {
            "ttfa_latency_speedup": round(ttfa_speedup, 2),
            "bandwidth_overhead_ratio": round(bandwidth_ratio, 2),
            "key_finding": (
                f"S2S delivers first audio {ttfa_speedup:.1f}x faster than S2T2S "
                f"({s2s_st_stats['mean_ttfa_ms']} ms vs {s2t2s_stats['mean_ttfa_ms']} ms), "
                f"while S2T2S achieves a {bandwidth_ratio:.1f}x bandwidth advantage "
                f"({s2t2s_stats['mean_bitrate_bps']} bps vs {s2s_st_stats['mean_bitrate_bps']} bps)."
            ),
        }

    # Breakdown by category (Short, Medium, Long) for S2T2S vs S2S
    categories = list(dict.fromkeys(r["category"] for r in results))
    for cat in categories:
        cat_s2t2s = [r for r in results if r["category"] == cat and r["pipeline"] == "S2T2S"]
        cat_s2s = [r for r in results if r["category"] == cat and r["pipeline"] == "S2S (SpeechTokenizer)"]

        summary["categories"][cat] = {
            "s2t2s_mean_ttfa_ms": round(sum(r["ttfa_ms"] for r in cat_s2t2s) / len(cat_s2t2s), 1) if cat_s2t2s else 0,
            "s2s_mean_ttfa_ms": round(sum(r["ttfa_ms"] for r in cat_s2s) / len(cat_s2s), 1) if cat_s2s else 0,
            "s2t2s_mean_bitrate": round(sum(r["bitrate_bps"] for r in cat_s2t2s) / len(cat_s2t2s), 1) if cat_s2t2s else 0,
            "s2s_mean_bitrate": round(sum(r["bitrate_bps"] for r in cat_s2s) / len(cat_s2s), 1) if cat_s2s else 0,
        }

    # Breakdown by SNR for S2T2S BLEU scores
    snrs = sorted(list(dict.fromkeys(r["snr_db"] for r in results)))
    for snr in snrs:
        snr_subset = [r for r in results if r["snr_db"] == snr and r["pipeline"] == "S2T2S"]
        if snr_subset:
            mean_bleu = sum(r["bleu_score"] for r in snr_subset) / len(snr_subset)
            summary["snr_fidelity"][f"{snr}_dB"] = round(mean_bleu, 4)

    return summary


def print_executive_summary(summary: Dict[str, Any]):
    """Print clean executive summary table to console."""
    print("\n" + "=" * 80)
    print("  EXECUTIVE SUMMARY: 2x2 BENCHMARK MATRIX (BANDWIDTH VS. LATENCY)")
    print("=" * 80)
    print(f"  {'Pipeline':<24} {'TTFA (ms)':<12} {'Bitrate (bps)':<16} {'RTF':<10} {'Advantage':<18}")
    print(f"  {'-'*24} {'-'*12} {'-'*16} {'-'*10} {'-'*18}")

    for p, stats in summary["pipelines"].items():
        if "S2T2S" in p:
            advantage = "Bandwidth Winner"
        elif "SpeechTokenizer" in p:
            advantage = "Latency Winner"
        else:
            advantage = "Streaming Codec"

        print(
            f"  {p:<24} "
            f"{stats['mean_ttfa_ms']:<12.1f} "
            f"{stats['mean_bitrate_bps']:<16.1f} "
            f"{stats['mean_rtf']:<10.3f} "
            f"{advantage:<18}"
        )

    print("-" * 80)
    if "comparison" in summary and "key_finding" in summary["comparison"]:
        print(f"\n  [KEY FINDING]")
        print(f"  {summary['comparison']['key_finding']}")
        print(f"  - TTFA Speedup Factor : {summary['comparison']['ttfa_latency_speedup']}x")
        print(f"  - Bandwidth Overhead  : {summary['comparison']['bandwidth_overhead_ratio']}x")

    print("\n  [CATEGORY BREAKDOWN: DURATION SCALING]")
    print(f"  {'Category':<10} {'S2T2S TTFA':<16} {'S2S TTFA':<16} {'S2T2S Bitrate':<16} {'S2S Bitrate':<16}")
    print(f"  {'-'*10} {'-'*16} {'-'*16} {'-'*16} {'-'*16}")
    for cat, data in summary.get("categories", {}).items():
        print(
            f"  {cat:<10} "
            f"{data['s2t2s_mean_ttfa_ms']:<16.1f} "
            f"{data['s2s_mean_ttfa_ms']:<16.1f} "
            f"{data['s2t2s_mean_bitrate']:<16.1f} "
            f"{data['s2s_mean_bitrate']:<16.1f}"
        )
    print("=" * 80 + "\n")


if __name__ == "__main__":
    cli_args = parse_args()
    run_benchmark(cli_args)
