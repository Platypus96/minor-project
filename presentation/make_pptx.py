"""
PowerPoint Slide Deck Generator (12 Widescreen Slides)
Bandwidth vs. Time Efficiency: Semantic S2S vs. S2T2S Pipelines

Builds a professional, conference-grade 16:9 .pptx presentation
with embedded publication charts and presenter speaker notes.
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHARTS_DIR = os.path.join(_PROJECT_ROOT, "outputs", "charts")
OUTPUT_PPTX = os.path.join(_PROJECT_ROOT, "presentation", "midsem_presentation.pptx")

try:
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN
    from pptx.enum.shapes import MSO_SHAPE
except ImportError:
    print("[Error] python-pptx is not installed. Please run: pip install python-pptx")
    sys.exit(1)


# Color Palette
COLOR_PRIMARY = RGBColor(30, 58, 138)     # Navy #1e3a8a
COLOR_SECONDARY = RGBColor(15, 23, 42)    # Slate dark #0f172a
COLOR_ACCENT_BLUE = RGBColor(59, 130, 246) # Blue #3b82f6
COLOR_ACCENT_GREEN = RGBColor(16, 185, 129) # Emerald #10b981
COLOR_ACCENT_RED = RGBColor(239, 68, 68)   # Red #ef4444
COLOR_MUTED = RGBColor(100, 116, 139)     # Slate #64748b
COLOR_WHITE = RGBColor(255, 255, 255)
COLOR_CARD_BG = RGBColor(248, 250, 252)


def add_slide_header(slide, title_text, category_tag):
    """Add a uniform, stylish header with title and category pill."""
    # Title
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(9.5), Inches(0.8))
    tf = title_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    # Category Tag
    tag_box = slide.shapes.add_textbox(Inches(10.5), Inches(0.4), Inches(2.0), Inches(0.5))
    tf_tag = tag_box.text_frame
    p_tag = tf_tag.paragraphs[0]
    p_tag.alignment = PP_ALIGN.RIGHT
    p_tag.text = category_tag.upper()
    p_tag.font.size = Pt(10)
    p_tag.font.bold = True
    p_tag.font.color.rgb = COLOR_ACCENT_BLUE


def add_bullet_point(text_frame, bold_prefix, text, level=0, pt_size=12):
    """Add a bullet point with bold prefix."""
    p = text_frame.add_paragraph()
    p.level = level
    p.space_after = Pt(6)
    
    run_bold = p.add_run()
    run_bold.text = bold_prefix + " "
    run_bold.font.bold = True
    run_bold.font.size = Pt(pt_size)
    run_bold.font.color.rgb = COLOR_SECONDARY

    run_text = p.add_run()
    run_text.text = text
    run_text.font.bold = False
    run_text.font.size = Pt(pt_size)
    run_text.font.color.rgb = COLOR_MUTED


def set_speaker_notes(slide, notes_text):
    """Attach presenter speech notes to slide."""
    notes_slide = slide.notes_slide
    text_frame = notes_slide.notes_text_frame
    text_frame.text = notes_text


def build_presentation():
    print("[PPTX] Building 12-slide master presentation...")
    prs = Presentation()
    
    # 16:9 Widescreen dimensions
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # ================================================================== #
    # Slide 1: Title Slide (Dark Navy Background)
    # ================================================================== #
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = COLOR_SECONDARY
    bg1.line.fill.background()

    tb = s1.shapes.add_textbox(Inches(1.2), Inches(1.5), Inches(11.0), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = "IIIT ALLAHABAD • MINOR PROJECT MIDSEM EVALUATION"
    p0.font.size = Pt(13)
    p0.font.bold = True
    p0.font.color.rgb = COLOR_ACCENT_GREEN
    p0.space_after = Pt(16)

    p1 = tf.add_paragraph()
    p1.text = "Bandwidth vs. Time Efficiency:\nSemantic S2S vs. S2T2S Pipelines"
    p1.font.size = Pt(34)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_WHITE
    p1.space_after = Pt(14)

    p2 = tf.add_paragraph()
    p2.text = "Proving the Fundamental Trade-Off in Neural Speech Communication"
    p2.font.size = Pt(18)
    p2.font.color.rgb = COLOR_MUTED
    p2.space_after = Pt(28)

    p3 = tf.add_paragraph()
    p3.text = "FunASR SenseVoice Small  •  DeepSC Transformer  •  Kokoro Emotion TTS  •  SpeechTokenizer"
    p3.font.size = Pt(12)
    p3.font.color.rgb = COLOR_ACCENT_BLUE

    set_speaker_notes(
        s1,
        "Welcome the committee. Introduce the research theme: Bandwidth vs Time Efficiency. "
        "State that this midsem evaluation directly answers the fundamental trade-off between "
        "direct semantic S2S and cascade S2T2S pipelines with empirical data."
    )

    # ================================================================== #
    # Slide 2: Motivation & The Paradox
    # ================================================================== #
    s2 = prs.slides.add_slide(blank_layout)
    add_slide_header(s2, "The Motivation: The Bandwidth vs. Latency Paradox", "Problem Formulation")

    # Left Box: Bandwidth
    shape_left = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(1.5), Inches(5.3), Inches(5.2))
    shape_left.fill.solid()
    shape_left.fill.fore_color.rgb = COLOR_CARD_BG
    shape_left.line.color.rgb = COLOR_ACCENT_BLUE
    tf_l = shape_left.text_frame
    tf_l.word_wrap = True
    p = tf_l.paragraphs[0]
    p.text = "The Bandwidth Perspective"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY
    p.space_after = Pt(14)
    add_bullet_point(tf_l, "Text as Ultimate Compression:", "Converting speech to text strips raw acoustics, dropping bitrate to ~100-150 bps (50x lower than G.711).")
    add_bullet_point(tf_l, "Shannon Capacity Limits:", "Spectrum is crowded. Semantic communication saves critical wireless channel uses.")
    add_bullet_point(tf_l, "The Common Premise:", "If text uses vastly fewer bits, shouldn't S2T2S be the undisputed winner?")

    # Right Box: Latency
    shape_right = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.0), Inches(1.5), Inches(5.3), Inches(5.2))
    shape_right.fill.solid()
    shape_right.fill.fore_color.rgb = COLOR_CARD_BG
    shape_right.line.color.rgb = COLOR_ACCENT_RED
    tf_r = shape_right.text_frame
    tf_r.word_wrap = True
    p = tf_r.paragraphs[0]
    p.text = "The Time Efficiency Perspective"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT_RED
    p.space_after = Pt(14)
    add_bullet_point(tf_r, "Transmission Delay is Gone:", "Over 5G or modern links, transmitting 500 bytes takes <3 ms. Network lag is negligible.")
    add_bullet_point(tf_r, "The Real Delay:", "Latency in neural pipelines is dominated by algorithmic buffering and model compute.")
    add_bullet_point(tf_r, "Human Interaction Limit:", "ITU-T G.114 mandates delay <200 ms for natural dialogue. Delays >400 ms break conversational flow.")

    set_speaker_notes(
        s2,
        "Highlight the core paradox: On modern networks, physical byte transmission is instant (<3 ms). "
        "However, S2T2S introduces huge delay (5-20 seconds). Contrast the bandwidth mindset with the "
        "latency mindset. Mention ITU-T G.114 standard (200 ms conversational limit)."
    )

    # ================================================================== #
    # Slide 3: Architectures Compared
    # ================================================================== #
    s3 = prs.slides.add_slide(blank_layout)
    add_slide_header(s3, "System Architectures: S2T2S vs. Direct Semantic S2S", "Architectures")

    # Pipeline B Box
    b_box = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(1.5), Inches(11.3), Inches(2.4))
    b_box.fill.solid()
    b_box.fill.fore_color.rgb = COLOR_CARD_BG
    b_box.line.color.rgb = COLOR_ACCENT_RED
    tf_b = b_box.text_frame
    tf_b.word_wrap = True
    p = tf_b.paragraphs[0]
    p.text = "Pipeline B: Cascade S2T2S (Implemented & Trained in Project)"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT_RED
    p.space_after = Pt(8)
    add_bullet_point(tf_b, "Flow:", "Audio In ➔ SenseVoice ASR (Text + Emotion/Gender) ➔ DeepSC Encoder ➔ Channel ➔ DeepSC Decoder ➔ Kokoro TTS ➔ Audio Out")
    add_bullet_point(tf_b, "Key Trait:", "Mandatory phrase buffering (1.5s - 25s) before DeepSC can encode words. 3-stage cascaded neural inference.")

    # Pipeline A Box
    a_box = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(4.3), Inches(11.3), Inches(2.4))
    a_box.fill.solid()
    a_box.fill.fore_color.rgb = COLOR_CARD_BG
    a_box.line.color.rgb = COLOR_ACCENT_GREEN
    tf_a = a_box.text_frame
    tf_a.word_wrap = True
    p = tf_a.paragraphs[0]
    p.text = "Pipeline A: Direct Semantic S2S (Streaming Audio Tokenizers)"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT_GREEN
    p.space_after = Pt(8)
    add_bullet_point(tf_a, "Flow:", "Audio In ➔ Streaming Neural Tokenizer (SpeechTokenizer / Mimi) ➔ Token Stream ➔ Channel ➔ Streaming Vocoder ➔ Audio Out")
    add_bullet_point(tf_a, "Key Trait:", "Operates on 20 ms frames. First audio is reconstructed as soon as the first frame arrives (TTFA < 65 ms).")

    set_speaker_notes(
        s3,
        "Walk through both architectures. In S2T2S, explain our trained DeepSC model with 22,234 vocabulary, "
        "trained across AWGN, Rayleigh, and Rician channels. In S2S, explain frame-by-frame RVQ neural codecs (SpeechTokenizer, Mimi)."
    )

    # ================================================================== #
    # Slide 4: Theoretical Delay Formulation
    # ================================================================== #
    s4 = prs.slides.add_slide(blank_layout)
    add_slide_header(s4, "Theoretical Delay Formulation: Mathematical Roots of Latency", "Mathematical Model")

    # Formula Box
    f_box = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(1.4), Inches(11.3), Inches(0.9))
    f_box.fill.solid()
    f_box.fill.fore_color.rgb = RGBColor(239, 246, 255)
    f_box.line.color.rgb = COLOR_ACCENT_BLUE
    tf_f = f_box.text_frame
    p = tf_f.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    p.text = "Latency(total) = T(buffering) + T(encode) + T(transmission) + T(decode)"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    # 2 Detail Boxes
    s4_l = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(2.5), Inches(5.4), Inches(4.3))
    s4_l.fill.solid()
    s4_l.fill.fore_color.rgb = COLOR_CARD_BG
    s4_l.line.color.rgb = COLOR_ACCENT_RED
    tf_4l = s4_l.text_frame
    tf_4l.word_wrap = True
    p = tf_4l.paragraphs[0]
    p.text = "1. The Context Bottleneck (Algorithmic)"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT_RED
    p.space_after = Pt(10)
    add_bullet_point(tf_4l, "ASR Clause Lookahead:", "ASR cannot transcribe token-by-token in real time. It must buffer an entire sentence to resolve phonemes and grammar: T_buf ≈ Audio Duration (1.5s - 25s).")
    add_bullet_point(tf_4l, "TTS Intonation Context:", "TTS must wait for the entire sentence to compute prosody before synthesizing sound.")

    s4_r = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.9), Inches(2.5), Inches(5.4), Inches(4.3))
    s4_r.fill.solid()
    s4_r.fill.fore_color.rgb = COLOR_CARD_BG
    s4_r.line.color.rgb = COLOR_PRIMARY
    tf_4r = s4_r.text_frame
    tf_4r.word_wrap = True
    p = tf_4r.paragraphs[0]
    p.text = "2. The Computational Cascade (Inference)"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY
    p.space_after = Pt(10)
    add_bullet_point(tf_4r, "Cascaded Pipeline:", "Runs 3 heavy models in series: SenseVoice ASR (300-1,200 ms) + DeepSC Transformer (20-250 ms) + Kokoro TTS (1,500-3,000 ms).")
    add_bullet_point(tf_4r, "S2S Frame Stream:", "Runs a single lightweight convolutional encoder (~5 ms/frame) and streaming vocoder (~15 ms/frame).")

    set_speaker_notes(
        s4,
        "Explain the mathematical deconstruction of delay. State why T_transmission is trivial (<3ms). "
        "Focus on the two key drivers: The Context Bottleneck (algorithmic lookahead) and the Computational Cascade (sequential inference)."
    )

    # ================================================================== #
    # Slide 5: Methodology & 2x2 Matrix
    # ================================================================== #
    s5 = prs.slides.add_slide(blank_layout)
    add_slide_header(s5, "Experimental Setup & The 2×2 Benchmark Matrix", "Methodology")

    t_box = s5.shapes.add_textbox(Inches(1.0), Inches(1.3), Inches(11.3), Inches(0.8))
    tf_t = t_box.text_frame
    p = tf_t.paragraphs[0]
    p.text = "Evaluated 72 automated experimental trials across 6 standardized audio samples, 3 wireless channels (AWGN, Rayleigh Fading, Rician Fading), and SNR ranging from -5 dB to +20 dB."
    p.font.size = Pt(13)
    p.font.color.rgb = COLOR_MUTED

    # Table
    rows, cols = 5, 4
    table_shape = s5.shapes.add_table(rows, cols, Inches(1.0), Inches(2.2), Inches(11.3), Inches(4.4))
    table = table_shape.table
    table.columns[0].width = Inches(2.8)
    table.columns[1].width = Inches(2.8)
    table.columns[2].width = Inches(2.8)
    table.columns[3].width = Inches(2.9)

    headers = ["Metric Dimension", "Pipeline A: S2S (SpeechTokenizer)", "Pipeline A: S2S (Kyutai Mimi)", "Pipeline B: Cascade S2T2S (DeepSC)"]
    for c_idx, h in enumerate(headers):
        cell = table.cell(0, c_idx)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_PRIMARY
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(11)
            p.font.bold = True
            p.font.color.rgb = COLOR_WHITE

    data = [
        ["Effective Bitrate", "1,500 bps", "1,100 bps", "622 - 1,077 bps (Winner)"],
        ["Time-to-First-Audio (TTFA)", "63.0 ms (Winner)", "43.0 ms (Winner)", "11,743.2 ms (11.7s avg)"],
        ["Algorithmic Lookahead", "20 ms Frame (Winner)", "12.5 ms Frame (Winner)", "1,590 to 24,685 ms (Whole clause)"],
        ["Streaming Usability", "Real-Time Conversational", "Real-Time Conversational", "Non-interactive / Store & Forward"],
    ]

    for r_idx, row_vals in enumerate(data):
        for c_idx, val in enumerate(row_vals):
            cell = table.cell(r_idx + 1, c_idx)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = COLOR_CARD_BG if r_idx % 2 == 0 else COLOR_WHITE
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(11)
                p.font.color.rgb = COLOR_SECONDARY
                if "Winner" in val:
                    p.font.bold = True
                    p.font.color.rgb = COLOR_ACCENT_GREEN if "63" in val or "43" in val or "20" in val or "12" in val else COLOR_PRIMARY

    set_speaker_notes(
        s5,
        "Present the experimental design. 6 standardized audio samples, 3 wireless channels, -5 to 20 dB SNR. "
        "Explain hardware-level instrumentation with time.perf_counter()."
    )

    # ================================================================== #
    # Slide 6: Chart 1 - Bandwidth Efficiency
    # ================================================================== #
    s6 = prs.slides.add_slide(blank_layout)
    add_slide_header(s6, "Empirical Result 1: Bandwidth Efficiency (S2T2S Wins)", "Data Rate (bps)")

    img_p1 = os.path.join(CHARTS_DIR, "bandwidth_comparison.png")
    if os.path.exists(img_p1):
        s6.shapes.add_picture(img_p1, Inches(0.8), Inches(1.5), Inches(6.8), Inches(5.2))

    tb_6 = s6.shapes.add_textbox(Inches(7.8), Inches(1.6), Inches(4.8), Inches(5.0))
    tf_6 = tb_6.text_frame
    tf_6.word_wrap = True
    p = tf_6.paragraphs[0]
    p.text = "Key Empirical Findings on Bandwidth:"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY
    p.space_after = Pt(12)
    add_bullet_point(tf_6, "Short Speech (<3s):", "S2T2S requires ~1,127 bps due to DeepSC start/end token symbols and 16-dim representation.")
    add_bullet_point(tf_6, "Paragraph Speech (>20s):", "S2T2S effective bitrate drops to 622 bps -- outperforming SpeechTokenizer (1,500 bps) by 2.4x and Mimi (1,100 bps) by 1.8x.")
    add_bullet_point(tf_6, "Text Compression Advantage:", "Text discards acoustic redundancy (timbre, room reverb), preserving purely linguistic semantics.")

    set_speaker_notes(
        s6,
        "Chart 1: Point to the long-duration paragraph bars. Note that S2T2S drops to 622 bps. "
        "Explain text semantic compression advantage."
    )

    # ================================================================== #
    # Slide 7: Chart 2 - TTFA Latency
    # ================================================================== #
    s7 = prs.slides.add_slide(blank_layout)
    add_slide_header(s7, "Empirical Result 2: Time-to-First-Audio (Semantic S2S Wins)", "Time Efficiency (TTFA)")

    img_p2 = os.path.join(CHARTS_DIR, "ttfa_comparison.png")
    if os.path.exists(img_p2):
        s7.shapes.add_picture(img_p2, Inches(0.8), Inches(1.5), Inches(6.8), Inches(5.2))

    tb_7 = s7.shapes.add_textbox(Inches(7.8), Inches(1.6), Inches(4.8), Inches(5.0))
    tf_7 = tb_7.text_frame
    tf_7.word_wrap = True
    p = tf_7.paragraphs[0]
    p.text = "Key Empirical Findings on Latency:"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT_GREEN
    p.space_after = Pt(12)
    add_bullet_point(tf_7, "ITU-T G.114 Standard:", "Humans perceive delay >200 ms as conversational disruption (amber dotted line).")
    add_bullet_point(tf_7, "S2S Real-Time Mastery:", "SpeechTokenizer reaches 63.0 ms; Kyutai Mimi reaches 43.0 ms -- both well within live conversation limits.")
    add_bullet_point(tf_7, "S2T2S Latency Explosion:", "Short greetings take 5,278 ms; paragraphs take 29,169 ms.")
    add_bullet_point(tf_7, "Speedup Factor:", "Semantic S2S delivers audio 186x to 273x faster than S2T2S!")

    set_speaker_notes(
        s7,
        "Chart 2: The critical contrast. Point out the amber 200ms line. S2S is at 43-63ms. "
        "S2T2S is at 5,000-29,000ms. A 186x to 273x latency disparity."
    )

    # ================================================================== #
    # Slide 8: Chart 3 - Context Bottleneck Proof
    # ================================================================== #
    s8 = prs.slides.add_slide(blank_layout)
    add_slide_header(s8, "Root Cause Analysis: Proving the Context Bottleneck", "Latency Deconstruction")

    img_p3 = os.path.join(CHARTS_DIR, "latency_breakdown.png")
    if os.path.exists(img_p3):
        s8.shapes.add_picture(img_p3, Inches(0.8), Inches(1.5), Inches(6.8), Inches(5.2))

    tb_8 = s8.shapes.add_textbox(Inches(7.8), Inches(1.6), Inches(4.8), Inches(5.0))
    tf_8 = tb_8.text_frame
    tf_8.word_wrap = True
    p = tf_8.paragraphs[0]
    p.text = "Deconstruction of a 7.25s Utterance:"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT_RED
    p.space_after = Pt(12)
    add_bullet_point(tf_8, "Phrase Buffering (Red):", "7,248 ms (65.6% of total delay) is spent simply waiting for the speaker to finish.")
    add_bullet_point(tf_8, "DeepSC Transmission (Blue):", "Only 175 ms (1.6%). Physical semantic communication is extremely fast.")
    add_bullet_point(tf_8, "TTS Synthesis (Purple):", "3,000 ms (27.1%) generating full sentence spectrograms.")
    add_bullet_point(tf_8, "The Definitive Proof:", "Even if computation was instantaneous (T_compute = 0), S2T2S still has >7 seconds of delay.")

    set_speaker_notes(
        s8,
        "Chart 3: Deconstruct the 7.25-second utterance. Phrase buffering is 7,248 ms (65.6%). "
        "Prove that even with instantaneous compute, S2T2S still has 7+ seconds of delay."
    )

    # ================================================================== #
    # Slide 9: Chart 4 - Scaling & RTF
    # ================================================================== #
    s9 = prs.slides.add_slide(blank_layout)
    add_slide_header(s9, "Scaling Behavior: TTFA vs. Duration & Throughput (RTF)", "Scaling Complexity")

    img_p4 = os.path.join(CHARTS_DIR, "rtf_vs_duration.png")
    if os.path.exists(img_p4):
        s9.shapes.add_picture(img_p4, Inches(0.8), Inches(1.5), Inches(6.8), Inches(5.2))

    tb_9 = s9.shapes.add_textbox(Inches(7.8), Inches(1.6), Inches(4.8), Inches(5.0))
    tf_9 = tb_9.text_frame
    tf_9.word_wrap = True
    p = tf_9.paragraphs[0]
    p.text = "Duration Scaling & RTF Nuance:"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY
    p.space_after = Pt(12)
    add_bullet_point(tf_9, "Linear Scaling O(N):", "S2T2S TTFA scales linearly with speech length (TTFA ≈ Duration + 4s).")
    add_bullet_point(tf_9, "Constant O(1) Streaming:", "S2S TTFA is completely flat at 63 ms regardless of audio length.")
    add_bullet_point(tf_9, "The RTF Paradox:", "S2T2S has an RTF of 0.06 to 0.15 (16x faster than real-time batch compute). But high throughput cannot overcome the phrase buffer bottleneck.")

    set_speaker_notes(
        s9,
        "Chart 4: Left panel shows linear O(N) scaling for S2T2S vs flat O(1) for S2S. "
        "Right panel shows RTF. S2T2S has low compute RTF (0.06), but buffering prevents streaming."
    )

    # ================================================================== #
    # Slide 10: Chart 5 - The Pareto Frontier (The Money Slide)
    # ================================================================== #
    s10 = prs.slides.add_slide(blank_layout)
    add_slide_header(s10, "The Definitive Proof: The Empirical Pareto Frontier", "The Money Slide ⭐")

    img_p5 = os.path.join(CHARTS_DIR, "tradeoff_scatter.png")
    if os.path.exists(img_p5):
        s10.shapes.add_picture(img_p5, Inches(0.8), Inches(1.5), Inches(6.8), Inches(5.2))

    tb_10 = s10.shapes.add_textbox(Inches(7.8), Inches(1.6), Inches(4.8), Inches(5.0))
    tf_10 = tb_10.text_frame
    tf_10.word_wrap = True
    p = tf_10.paragraphs[0]
    p.text = "Mapping the Operating Regimes:"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY
    p.space_after = Pt(12)
    add_bullet_point(tf_10, "No Single Winner:", "Neither pipeline occupies the ideal bottom-left origin (low bitrate + low latency).")
    add_bullet_point(tf_10, "Conversational Zone (Green):", "S2S (43-63 ms, 1,100-1,500 bps). Mandatory for interactive calls and conversational AI.")
    add_bullet_point(tf_10, "Bandwidth-Constrained Zone (Blue):", "S2T2S (622-1,000 bps, 5-29s). Optimal for satellite telemetry, voicemail, and sensor logs.")

    set_speaker_notes(
        s10,
        "Chart 5: THE MONEY SLIDE. Walk through the Pareto Frontier. Explain the two distinct operational zones: "
        "Conversational Streaming Zone vs Bandwidth Archival Zone. State the core engineering takeaway."
    )

    # ================================================================== #
    # Slide 11: Wireless Robustness
    # ================================================================== #
    s11 = prs.slides.add_slide(blank_layout)
    add_slide_header(s11, "Physical-Layer Wireless Robustness: DeepSC Under Fading", "Channel Fidelity")

    img_p6 = os.path.join(CHARTS_DIR, "snr_vs_fidelity.png")
    if os.path.exists(img_p6):
        s11.shapes.add_picture(img_p6, Inches(0.8), Inches(1.5), Inches(6.8), Inches(5.2))

    tb_11 = s11.shapes.add_textbox(Inches(7.8), Inches(1.6), Inches(4.8), Inches(5.0))
    tf_11 = tb_11.text_frame
    tf_11.word_wrap = True
    p = tf_11.paragraphs[0]
    p.text = "DeepSC Channel Performance:"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY
    p.space_after = Pt(12)
    add_bullet_point(tf_11, "Tested Fading Channels:", "AWGN, Rayleigh Fading (multipath), and Rician Fading (line-of-sight).")
    add_bullet_point(tf_11, "Graceful Degradation:", "DeepSC maintains semantic BLEU fidelity even at 0 dB without the cliff-edge drop seen in traditional 16-QAM + LDPC systems.")
    add_bullet_point(tf_11, "Equalization:", "Trained complex channel matrices apply zero-forcing equalization directly in neural embedding space.")

    set_speaker_notes(
        s11,
        "Chart 6: Show DeepSC wireless channel robustness. Explain that our model was trained on Europarl "
        "and handles AWGN, Rayleigh fading, and Rician fading without cliff effect."
    )

    # ================================================================== #
    # Slide 12: Conclusion & Defense Summary
    # ================================================================== #
    s12 = prs.slides.add_slide(blank_layout)
    add_slide_header(s12, "Engineering Conclusion & Midsem Defense Summary", "Conclusion")

    # Conclusion Card Left
    cl = s12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(1.5), Inches(5.4), Inches(5.2))
    cl.fill.solid()
    cl.fill.fore_color.rgb = COLOR_CARD_BG
    cl.line.color.rgb = COLOR_PRIMARY
    tf_cl = cl.text_frame
    tf_cl.word_wrap = True
    p = tf_cl.paragraphs[0]
    p.text = "The Proven Engineering Thesis"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY
    p.space_after = Pt(10)
    add_bullet_point(tf_cl, "The Conclusion:", "'Converting speech to text yields up to 2.4x reduction in bandwidth, but incurs a 186x to 273x latency penalty due to phrase buffering and cascading inference.'")
    add_bullet_point(tf_cl, "Takeaway:", "Direct Semantic S2S is not optimized for raw compression, but for conversational flow.")
    add_bullet_point(tf_cl, "Midsem Deliverables:", "Complete pipeline working, 22k vocab model loaded, 72 benchmark trials, 6 publication figures.")

    # Roadmap Card Right
    cr = s12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.9), Inches(1.5), Inches(5.4), Inches(5.2))
    cr.fill.solid()
    cr.fill.fore_color.rgb = COLOR_CARD_BG
    cr.line.color.rgb = COLOR_ACCENT_GREEN
    tf_cr = cr.text_frame
    tf_cr.word_wrap = True
    p = tf_cr.paragraphs[0]
    p.text = "Live Demo & Endsem Roadmap"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT_GREEN
    p.space_after = Pt(10)
    add_bullet_point(tf_cr, "Interactive Demo Ready:", "Gradio UI running (run_ui.bat) with mic input, channel fading controls, and live constellation scatter plotting.")
    add_bullet_point(tf_cr, "Endsem Roadmap 1:", "Integrate native streaming Mimi/SpeechTokenizer encoder weights directly into DeepSC channel loop.")
    add_bullet_point(tf_cr, "Endsem Roadmap 2:", "Hardware-in-the-loop over-the-air validation with SDR (HackRF / USRP).")
    add_bullet_point(tf_cr, "Endsem Roadmap 3:", "Acoustic evaluation with PESQ and STOI metrics.")

    set_speaker_notes(
        s12,
        "To conclude: our midsem evaluation has proven the trade-off conclusively. "
        "S2T2S achieves superior bandwidth efficiency (622 bps) at the cost of a 186-fold latency penalty. "
        "Semantic S2S operates frame-by-frame, delivering first audio in 43-63 ms to preserve real-time conversational flow. "
        "Thank the committee and open for Q&A and live demo."
    )

    prs.save(OUTPUT_PPTX)
    print(f"[PPTX] Successfully generated 12-slide presentation at: {OUTPUT_PPTX}")


if __name__ == "__main__":
    build_presentation()
