"""
Gradio Web UI for Semantic Speech Communication.

Tabs:
  1. Pipeline Demo      — full speech-to-speech with live stepper
  2. Channel Comparison — same text through AWGN / Rayleigh / Rician
  3. Performance Analysis — SNR sweep curves + bandwidth dashboard
"""

import os
import sys
import base64

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import gradio as gr
from pipeline.pipeline import SemanticSpeechPipeline
from pipeline.translation import LANG_CHOICES

# ------------------------------------------------------------------ #
#  Global pipeline (loaded once)                                       #
# ------------------------------------------------------------------ #
print("Initializing pipeline …")
pipeline = SemanticSpeechPipeline()
print("Pipeline ready!\n")

try:
    from deep_translator import GoogleTranslator  # noqa: F401
    _TRANS = True
except ImportError:
    _TRANS = False


# ------------------------------------------------------------------ #
#  Logo — embed as base64 so it always loads regardless of path        #
# ------------------------------------------------------------------ #
def _logo_data_uri():
    path = os.path.join(_PROJECT_ROOT, "pipeline", "iiita_logo.png")
    if os.path.exists(path):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{b64}"
    return ""

LOGO_URI = _logo_data_uri()


# ------------------------------------------------------------------ #
#  HTML builders                                                       #
# ------------------------------------------------------------------ #

def header_html():
    logo = f'<img src="{LOGO_URI}" class="ssc-logo" alt="IIITA">' if LOGO_URI else ""
    return f"""
    <div class="ssc-header">
        {logo}
        <div class="ssc-header-text">
            <h1>Semantic Speech Communication</h1>
            <p>End-to-end speech semantic transmission over wireless channels</p>
        </div>
        <div class="ssc-badge">IIIT Allahabad &nbsp;&middot;&nbsp; Minor Project</div>
    </div>
    """


def stepper_html(statuses=None):
    if statuses is None:
        statuses = ["pending"] * 4

    steps = [
        ("Speech Recognition",  "1"),
        ("Semantic Encode",     "2"),
        ("Wireless Channel",    "3"),
        ("Speech Synthesis",    "4"),
    ]

    done_count = sum(1 for s in statuses if s == "done")
    running_idx = next((i for i, s in enumerate(statuses) if s == "running"), -1)

    if done_count == 4:
        fill_pct = 100
    elif running_idx >= 0:
        fill_pct = int((running_idx / 3) * 100)
    else:
        fill_pct = int((done_count / 3) * 100) if done_count > 0 else 0

    step_html = ""
    for i, (label, num) in enumerate(steps):
        cls = statuses[i] if statuses[i] in ("active", "done") else ""
        if statuses[i] == "running":
            cls = "active"
        icon = "&#10003;" if statuses[i] == "done" else num
        step_html += f"""
        <div class="ssc-step {cls}">
            <div class="ssc-step-circle">{icon}</div>
            <div class="ssc-step-label">{label}</div>
        </div>"""

    return f"""
    <div class="ssc-stepper">
        <div class="ssc-step-track">
            <div class="ssc-step-fill" style="width:{fill_pct}%"></div>
        </div>
        {step_html}
    </div>
    """


def bleu_bar_html(bleu: float) -> str:
    pct = min(100, int(bleu * 100))
    return f"""
    <div class="bleu-wrap">
        <div class="bleu-row">
            <span>BLEU Score</span>
            <span>{bleu:.4f}</span>
        </div>
        <div class="bleu-track">
            <div class="bleu-fill" style="width:{pct}%"></div>
        </div>
    </div>
    """


def channel_card_html(ch_name: str, bleu: float, recon: str) -> str:
    if bleu >= 0.7:
        cls = "high"
    elif bleu >= 0.35:
        cls = "mid"
    else:
        cls = "low"
    return f"""
    <div class="ch-card">
        <div style="display:flex;justify-content:space-between;align-items:center">
            <span class="ch-name">{ch_name}</span>
            <span class="ch-bleu {cls}">BLEU&nbsp;{bleu:.4f}</span>
        </div>
        <div class="ch-text">{recon}</div>
    </div>
    """


# ------------------------------------------------------------------ #
#  Tab 1 — Pipeline streaming                                          #
# ------------------------------------------------------------------ #

def run_pipeline(audio_path, channel_type, snr_db, language):
    if audio_path is None:
        yield {status_box: "Please upload or record an audio file."}
        return

    statuses = ["pending"] * 4
    yield {
        stepper: stepper_html(statuses),
        step1_group: gr.update(visible=False),
        step2_group: gr.update(visible=False),
        step3_group: gr.update(visible=False),
        status_box: "Initializing …",
    }

    out_dir = os.path.join(_PROJECT_ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "output.wav")

    try:
        for state in pipeline.run_streaming(
            audio_path, output_path, channel_type, float(snr_db), language
        ):
            p_step = state["step"]
            status  = state["status"]

            if status == "running":
                idx_map = {1: 0, 2: 1, 3: 3}
                si = idx_map.get(p_step, 0)
                statuses[si] = "running"
                yield {
                    stepper: stepper_html(statuses),
                    status_box: f"Step {p_step}: {state['label']} …",
                }

            elif status == "done":
                updates = {}
                if p_step == 1:
                    statuses[0] = "done"
                    detected = state.get("detected_lang", "en")
                    lang_label = next(
                        (lbl for lbl, code in LANG_CHOICES if code == detected), detected
                    )
                    updates[step1_group] = gr.update(visible=True)
                    updates[transcript_box] = state["text"]
                    updates[lang_badge] = (
                        f"<span class='badge badge-lang'>Language: {lang_label}</span>"
                        f"<span class='badge badge-emotion'>{state['emotion']}</span>"
                        f"<span class='badge badge-lang'>{state['gender']}</span>"
                    )
                    text_en = state.get("text_en", state["text"])
                    updates[trans_box] = (
                        gr.update(visible=True, value=f"English for DeepSC: {text_en}")
                        if text_en != state["text"]
                        else gr.update(visible=False)
                    )

                elif p_step == 2:
                    statuses[1] = "done"
                    statuses[2] = "done"
                    updates[step2_group] = gr.update(visible=True)
                    updates[orig_box]  = state["original_text"]
                    updates[recon_box] = state["reconstructed_text"]
                    orig_en  = state.get("original_text_en",     state["original_text"])
                    recon_en = state.get("reconstructed_text_en", state["reconstructed_text"])
                    updates[orig_en_box]  = (
                        gr.update(visible=True, value=f"EN: {orig_en}")
                        if orig_en != state["original_text"] else gr.update(visible=False)
                    )
                    updates[recon_en_box] = (
                        gr.update(visible=True, value=f"EN: {recon_en}")
                        if recon_en != state["reconstructed_text"] else gr.update(visible=False)
                    )
                    updates[bleu_html]   = bleu_bar_html(state["bleu_score"])
                    updates[ch_info_box] = (
                        f"<div class='channel-pill'>"
                        f"Channel: {state['channel']} &nbsp;|&nbsp; "
                        f"{state['snr_db']} dB &nbsp;|&nbsp; "
                        f"Mode: {state['deepsc_mode']}"
                        f"</div>"
                    )
                    cpath = state.get("constellation_path")
                    updates[const_img] = cpath if cpath and os.path.exists(cpath) else None

                elif p_step == 3:
                    statuses[3] = "done"
                    updates[step3_group] = gr.update(visible=True)
                    updates[audio_out]   = state["output_audio"]
                    updates[status_box]  = "Pipeline complete."

                updates[stepper] = stepper_html(statuses)
                yield updates

    except Exception as e:
        import traceback; traceback.print_exc()
        yield {status_box: f"Error: {e}"}


# ------------------------------------------------------------------ #
#  Tab 2 — Channel Comparison                                          #
# ------------------------------------------------------------------ #

def run_comparison(text_in, snr_db):
    if not text_in or not text_in.strip():
        return ("", "", "", "", "", None, "Please enter text.")

    from pipeline.comparison import compare_channels
    from pipeline.semantic.constellation import plot_constellation_comparison

    results = compare_channels(text_in.strip(), float(snr_db), pipeline.deepsc)

    cards = {}
    for ch in ["AWGN", "RAYLEIGH", "RICIAN"]:
        d = results.get(ch, {})
        cards[ch] = channel_card_html(ch, d.get("bleu_score", 0), d.get("reconstructed_text", "N/A"))

    # Constellation comparison plot
    cpath = None
    try:
        sym_data = {
            ch: {"pre_symbols": results[ch]["pre_symbols"], "post_symbols": results[ch]["post_symbols"]}
            for ch in results
            if len(results[ch].get("pre_symbols", [])) > 0
        }
        if sym_data:
            odir = os.path.join(_PROJECT_ROOT, "outputs")
            os.makedirs(odir, exist_ok=True)
            cpath = plot_constellation_comparison(
                sym_data, snr_db=float(snr_db),
                output_path=os.path.join(odir, "constellation_comparison.png"),
            )
    except Exception as e:
        print(f"Constellation error: {e}")

    best = max(results, key=lambda c: results[c]["bleu_score"])
    summary = f"**Best channel at {snr_db} dB:** {best} (BLEU {results[best]['bleu_score']:.4f})"

    return (
        summary,
        cards["AWGN"],
        cards["RAYLEIGH"],
        cards["RICIAN"],
        f"Input: {text_in.strip()}",
        cpath,
        "Comparison complete.",
    )


# ------------------------------------------------------------------ #
#  Tab 3 — Performance Analysis                                        #
# ------------------------------------------------------------------ #

def run_analysis(text_in, channel_sel):
    if not text_in or not text_in.strip():
        return (None, "", "Please enter text.")

    from pipeline.analysis import snr_sweep, plot_snr_curve
    from pipeline.bandwidth import compute_bandwidth_stats, format_bandwidth_html

    channels = ["AWGN", "RAYLEIGH", "RICIAN"] if channel_sel == "All Channels" else [channel_sel.upper()]

    odir = os.path.join(_PROJECT_ROOT, "outputs")
    os.makedirs(odir, exist_ok=True)

    sweep = snr_sweep(text_in.strip(), pipeline.deepsc, channel_types=channels)
    img   = plot_snr_curve(sweep, output_path=os.path.join(odir, "snr_curve.png"))

    stats = compute_bandwidth_stats(text_in.strip(), pipeline.deepsc)
    bw_html = format_bandwidth_html(stats)

    return (img, bw_html, "Analysis complete.")


# ------------------------------------------------------------------ #
#  Layout                                                              #
# ------------------------------------------------------------------ #

css_path = os.path.join(_PROJECT_ROOT, "pipeline", "static", "custom_theme.css")
try:
    with open(css_path, "r", encoding="utf-8") as f:
        custom_css = f.read()
except FileNotFoundError:
    custom_css = ""

# Use Base theme — we own all the styling ourselves
theme = gr.themes.Base(
    primary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
)

force_light_js = """
function() {
    document.body.classList.remove('dark');
    document.documentElement.classList.remove('dark');
    document.documentElement.style.colorScheme = 'light';
}
"""

with gr.Blocks(
    title="Semantic Speech Communication",
    theme=theme,
    css=custom_css,
    js=force_light_js,
) as demo:

    gr.HTML(header_html())

    with gr.Tabs():

        # ==================================================== #
        #  Tab 1: Pipeline Demo                                 #
        # ==================================================== #
        with gr.TabItem("Pipeline Demo"):

            stepper = gr.HTML(stepper_html())

            with gr.Row(equal_height=False):

                # Sidebar
                with gr.Column(scale=1, min_width=280):
                    with gr.Group(elem_classes="ssc-sidebar"):
                        gr.Markdown("**Input Audio**")
                        audio_in = gr.Audio(
                            type="filepath",
                            label="Upload or Record Speech",
                            sources=["upload", "microphone"],
                        )

                        gr.Markdown("**Language**")
                        lang_dd = gr.Dropdown(
                            choices=[(lbl, code) for lbl, code in LANG_CHOICES],
                            value="auto",
                            label="Input Language",
                        )

                        gr.Markdown("**Channel**")
                        ch_dd = gr.Dropdown(
                            choices=["AWGN", "RAYLEIGH", "RICIAN"],
                            value="AWGN",
                            label="Channel Type",
                        )
                        snr_sl = gr.Slider(
                            minimum=-5, maximum=20, value=10, step=1,
                            label="SNR (dB)",
                        )

                        run_btn = gr.Button("Run Pipeline", elem_classes="primary", size="lg")
                        status_box = gr.Textbox(
                            label="Status", interactive=False, value="Ready"
                        )

                # Results
                with gr.Column(scale=2):

                    with gr.Group(visible=False, elem_classes="ssc-result-card") as step1_group:
                        gr.Markdown("**Step 1 — Speech Recognition**")
                        lang_badge  = gr.HTML()
                        transcript_box = gr.Textbox(label="Transcript", interactive=False)
                        trans_box   = gr.Textbox(
                            label="English (for DeepSC)",
                            interactive=False, visible=False,
                        )

                    with gr.Group(visible=False, elem_classes="ssc-result-card") as step2_group:
                        gr.Markdown("**Step 2 — Semantic Transmission**")
                        ch_info_box = gr.HTML()
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("#### Original")
                                orig_box    = gr.Textbox(show_label=False, interactive=False, lines=2)
                                orig_en_box = gr.Textbox(show_label=False, interactive=False, lines=1, visible=False)
                            with gr.Column():
                                gr.Markdown("#### Reconstructed")
                                recon_box    = gr.Textbox(show_label=False, interactive=False, lines=2)
                                recon_en_box = gr.Textbox(show_label=False, interactive=False, lines=1, visible=False)
                        bleu_html = gr.HTML()
                        with gr.Accordion("Signal Constellation Diagram", open=False):
                            const_img = gr.Image(
                                label="I/Q Constellation",
                                interactive=False, type="filepath",
                            )

                    with gr.Group(visible=False, elem_classes="ssc-result-card") as step3_group:
                        gr.Markdown("**Step 3 — Emotion-Aware Speech Synthesis**")
                        audio_out = gr.Audio(label="Reconstructed Speech", interactive=False)

            run_btn.click(
                fn=run_pipeline,
                inputs=[audio_in, ch_dd, snr_sl, lang_dd],
                outputs=[
                    stepper,
                    step1_group, step2_group, step3_group,
                    transcript_box, trans_box,
                    lang_badge,
                    orig_box, orig_en_box,
                    recon_box, recon_en_box,
                    bleu_html, ch_info_box,
                    audio_out, status_box,
                    const_img,
                ],
            )

        # ==================================================== #
        #  Tab 2: Channel Comparison                            #
        # ==================================================== #
        with gr.TabItem("Channel Comparison"):

            gr.HTML("""
            <div class="ssc-info">
                <strong>Multi-Channel Comparison</strong> &mdash;
                Enter text and see how it performs across AWGN, Rayleigh, and Rician channels simultaneously.
                This answers the question: which wireless environment preserves semantic meaning best?
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=1, min_width=280):
                    with gr.Group(elem_classes="ssc-sidebar"):
                        cmp_text = gr.Textbox(
                            label="Text to transmit",
                            placeholder="Enter English text to compare across channels …",
                            lines=4,
                        )
                        cmp_snr = gr.Slider(minimum=-5, maximum=20, value=10, step=1, label="SNR (dB)")
                        cmp_btn = gr.Button("Compare Channels", elem_classes="primary", size="lg")
                        cmp_status = gr.Textbox(label="Status", interactive=False, value="Ready")

                with gr.Column(scale=2):
                    with gr.Group(elem_classes="ssc-result-card"):
                        gr.Markdown("**Channel Comparison Results**")
                        cmp_summary = gr.Markdown()
                        cmp_orig    = gr.Markdown()
                        
                        with gr.Row():
                            awgn_card     = gr.HTML()
                            rayleigh_card = gr.HTML()
                            rician_card   = gr.HTML()
                            
                        gr.Markdown("**Constellation Diagrams**")
                        cmp_const = gr.Image(
                            label="I/Q Constellation per Channel",
                            interactive=False, type="filepath",
                        )

            cmp_btn.click(
                fn=run_comparison,
                inputs=[cmp_text, cmp_snr],
                outputs=[
                    cmp_summary,
                    awgn_card, rayleigh_card, rician_card,
                    cmp_orig,
                    cmp_const,
                    cmp_status,
                ],
            )

        # ==================================================== #
        #  Tab 3: Performance Analysis                          #
        # ==================================================== #
        with gr.TabItem("Performance Analysis"):

            gr.HTML("""
            <div class="ssc-info">
                <strong>SNR Sweep &amp; Bandwidth Analysis</strong> &mdash;
                Generate the BLEU-vs-SNR performance curve and compare bandwidth usage of
                DeepSC versus traditional source+channel coding.
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=1, min_width=280):
                    with gr.Group(elem_classes="ssc-sidebar"):
                        ana_text = gr.Textbox(
                            label="Text to analyze",
                            placeholder="Enter English text for analysis …",
                            lines=4,
                        )
                        ana_ch = gr.Dropdown(
                            choices=["All Channels", "AWGN", "RAYLEIGH", "RICIAN"],
                            value="All Channels",
                            label="Channel(s)",
                        )
                        ana_btn = gr.Button("Run Analysis", elem_classes="primary", size="lg")
                        ana_status = gr.Textbox(label="Status", interactive=False, value="Ready")

                with gr.Column(scale=2):
                    with gr.Group(elem_classes="ssc-result-card"):
                        gr.Markdown("**BLEU Score vs. SNR (dB)**")
                        snr_img = gr.Image(
                            label="Performance Curve",
                            interactive=False, type="filepath",
                        )
                    
                    with gr.Group(elem_classes="ssc-result-card"):
                        gr.Markdown("**Bandwidth Compression Dashboard**")
                        bw_out = gr.HTML()

            ana_btn.click(
                fn=run_analysis,
                inputs=[ana_text, ana_ch],
                outputs=[snr_img, bw_out, ana_status],
            )

    gr.HTML("<div class='ssc-footer'>Semantic Speech Communication &nbsp;&middot;&nbsp; DeepSC (PyTorch) &nbsp;&middot;&nbsp; SenseVoice &nbsp;&middot;&nbsp; IIIT Allahabad</div>")

demo.allowed_paths = [_PROJECT_ROOT]

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )
