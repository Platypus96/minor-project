"""
Gradio Web UI for Semantic Speech Communication Demo.

Run:  python demo.py
Then open http://localhost:7860 in your browser.
"""

import os
import sys
import tempfile

# Project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import gradio as gr
from pipeline.pipeline import SemanticSpeechPipeline

# ------------------------------------------------------------------ #
#  Global pipeline instance (loaded once)                              #
# ------------------------------------------------------------------ #
print("Initializing pipeline — this may take a minute on first run...")
pipeline = SemanticSpeechPipeline()
print("Pipeline ready!\n")


# ------------------------------------------------------------------ #
#  Processing function                                                 #
# ------------------------------------------------------------------ #
def process_audio(audio_path, channel_type, snr_db):
    """Called when user clicks 'Run Pipeline'."""
    if audio_path is None:
        return "⚠ Please upload or record an audio file.", "", "", "", None, ""

    # Create a temp output file
    out_dir = os.path.join(_PROJECT_ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "output.wav")

    try:
        result = pipeline.run(
            audio_input_path=audio_path,
            output_path=output_path,
            channel=channel_type,
            snr_db=float(snr_db),
        )

        status = (
            f"✅ Pipeline complete! (DeepSC mode: {result['deepsc_mode']})\n"
            f"Channel: {result['channel']} @ SNR {result['snr_db']} dB"
        )
        return (
            result["original_text"],
            result["emotion"],
            result["gender"],
            result["reconstructed_text"],
            result["output_audio"],
            status,
            f"{result['bleu_score']:.4f}",
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return (
            f"Error: {str(e)}",
            "",
            "",
            "",
            None,
            f"❌ Error: {str(e)}",
            "0.0",
        )


# ------------------------------------------------------------------ #
#  Gradio UI                                                           #
# ------------------------------------------------------------------ #
with gr.Blocks(title="Semantic Speech Communication", theme=gr.themes.Default()) as demo:

    # --- Header ---
    with gr.Row():
        gr.Image(
            "pipeline/iiita_logo.png",
            show_label=False,
            container=False,
            height=100,
            scale=1
        )
        with gr.Column(scale=9):
            gr.Markdown("# Semantic Speech Communication")
            gr.Markdown("End-to-end speech semantic communication over wireless channels")
            gr.Markdown("> Speech → Text + Emotion + Gender → Semantic Encode → Channel → Decode → TTS")

    with gr.Row():
        # ---- LEFT COLUMN: Inputs ----
        with gr.Column(scale=1):
            gr.Markdown("### Input")
            audio_input = gr.Audio(
                type="filepath",
                label="Upload or Record Speech",
                sources=["upload", "microphone"],
            )

            gr.Markdown("### Channel Settings")
            channel_type = gr.Dropdown(
                choices=["AWGN", "RAYLEIGH", "RICIAN"],
                value="AWGN",
                label="Channel Type",
            )
            snr_slider = gr.Slider(
                minimum=-5,
                maximum=20,
                value=10,
                step=1,
                label="SNR (dB) — lower = noisier",
            )

            run_btn = gr.Button(
                "▶  Run Pipeline",
                variant="primary",
                size="lg",
            )

        # ---- RIGHT COLUMN: Outputs ----
        with gr.Column(scale=1):
            gr.Markdown("### Results")
            with gr.Row():
                emotion_out = gr.Textbox(label="Emotion", interactive=False)
                gender_out = gr.Textbox(label="Gender", interactive=False)
                bleu_out = gr.Textbox(label="BLEU Score", interactive=False)

            original_text = gr.Textbox(
                label="Original Text (from speech)",
                lines=3,
                interactive=False,
            )
            reconstructed_text = gr.Textbox(
                label="Reconstructed Text (after channel)",
                lines=3,
                interactive=False,
            )
            audio_output = gr.Audio(label="Output Speech", type="filepath")
            status_box = gr.Textbox(label="Status", interactive=False)

    # ---- Wire up ----
    run_btn.click(
        fn=process_audio,
        inputs=[audio_input, channel_type, snr_slider],
        outputs=[
            original_text,
            emotion_out,
            gender_out,
            reconstructed_text,
            audio_output,
            status_box,
            bleu_out,
        ],
    )

    # ---- Footer ----
    gr.Markdown(
        "---\n"
        "*Semantic Speech Communication Project* · "
        "DeepSC (PyTorch) · SenseVoice · TTS.ai API · "
        "AWGN / Rayleigh / Rician channels"
    )


# ------------------------------------------------------------------ #
#  Launch                                                              #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )
