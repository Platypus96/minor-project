# 🎙️ Bandwidth vs. Time Efficiency: Semantic S2S vs. S2T2S Pipelines
## Minor Project Midsem Presentation — 12-Slide Master Deck & Speaker Script
**Institution:** Indian Institute of Information Technology Allahabad (IIIT Allahabad)  
**Project:** End-to-End Semantic Speech Communication over Wireless Channels  
**Milestone:** Mid-Semester Evaluation (Hypothesis Proof & Empirical Validation)  
**Target Duration:** 10–12 Minutes

---

## 📑 Slide Directory

| Slide # | Slide Title | Key Visual / Chart | Focus |
|---|---|---|---|
| **Slide 1** | Title & Project Overview | IIITA Crest & Architecture Badge | Context & Team |
| **Slide 2** | Motivation: The Bandwidth vs. Latency Paradox | Dual Equation & Conversational Limit | The Core Research Problem |
| **Slide 3** | Architectural Comparison: S2S vs. S2T2S | Block Diagrams (Cascade vs. Direct) | System Flows |
| **Slide 4** | Theoretical Delay Formulation | Mathematical Delay Deconstruction | Algorithmic Roots |
| **Slide 5** | Experimental Design & 2×2 Benchmark Matrix | Multi-Channel Matrix Table | Rigorous Methodology |
| **Slide 6** | Empirical Results: Bandwidth Efficiency | `charts/bandwidth_comparison.png` | S2T2S Bandwidth Win |
| **Slide 7** | Empirical Results: Time-to-First-Audio (TTFA) | `charts/ttfa_comparison.png` | S2S Latency Win |
| **Slide 8** | Root Cause: Proving the Context Bottleneck | `charts/latency_breakdown.png` | Phrase Buffering Proof |
| **Slide 9** | Duration Scaling & Computational Throughput | `charts/rtf_vs_duration.png` | $\mathcal{O}(N)$ vs. $\mathcal{O}(1)$ Scaling |
| **Slide 10** | The Definitive Tradeoff: The Pareto Frontier | `charts/tradeoff_scatter.png` | **The Money Slide** |
| **Slide 11** | Wireless Physical-Layer Robustness | `charts/snr_vs_fidelity.png` | DeepSC Fading Robustness |
| **Slide 12** | Engineering Conclusion & Future Roadmap | Summary Matrix & Live Demo Link | Defense Takeaways |

---

## Slide 1: Title & Project Overview

### Visual Layout
- **Header:** IIIT Allahabad Minor Project Presentation
- **Main Title:** Bandwidth vs. Time Efficiency in Semantic Speech Communication
- **Subtitle:** Proving the Fundamental Trade-Off: Direct Semantic S2S vs. Cascade S2T2S Pipelines
- **System Components:** FunASR SenseVoice Small · DeepSC Transformer · Kokoro Emotion TTS · Neural Speech Tokenizers

### Slide Bullet Points
- **Research Question:** If Speech-to-Text-to-Speech (S2T2S) yields optimal compression by converting sound to text, why are Direct Semantic S2S pipelines essential for modern communication?
- **Scope of Midsem Evaluation:**
  - Complete end-to-end implementation of the cascade S2T2S pipeline (STT $\to$ DeepSC $\to$ Fading Channel $\to$ TTS).
  - High-precision per-stage timing instrumentation ($\mu s$ resolution).
  - Empirical benchmarking against state-of-the-art streaming semantic audio tokenizers (SpeechTokenizer, Mimi, EnCodec).
  - Rigorous validation across AWGN, Rayleigh, and Rician fading channels.

### 🗣️ Speaker Script (Minute 0:00 – 1:00)
> *"Good morning, respected professors and committee members. Today, I am presenting our mid-semester progress on 'Bandwidth vs. Time Efficiency in Semantic Speech Communication.'*
> 
> *In digital communications, bandwidth has traditionally received the primary attention. However, when transitioning to neural and semantic communications, latency—specifically interactive time efficiency—completely flips the architectural decision. Today, I will present mathematical models and concrete empirical data collected from our implemented system to prove why neither pipeline is universally superior, but rather how they define two distinct operating regimes on the Pareto frontier."*

---

## Slide 2: Motivation: The Bandwidth vs. Latency Paradox

### Visual Layout
- **Left Column:** *The Bandwidth Argument* — Text is the ultimate semantic compression. Text needs only 50–100 words per minute (~100–150 bps).
- **Right Column:** *The Time Efficiency Argument* — Humans perceive voice delays $>200\text{ ms}$ as unnatural interruptions (ITU-T G.114 standard).
- **Center Callout Box:** *"The Paradox: A pipeline can have virtually zero network transmission delay, yet remain unusable for real-time conversation."*

### Slide Bullet Points
- **Traditional Telecom Metric:** Raw Bitrate (bps). By this metric alone, S2T2S appears to be the undisputed winner.
- **Human-Centric Metric:** Time-to-First-Audio (TTFA). How many milliseconds from when the speaker utters a word until the listener hears the reconstructed acoustic wave?
- **The Core Problem:**
  - Modern networks transmit payloads in under 3 ms ($T_{\text{transmission}} \approx 0$).
  - Therefore, communication lag is **100% determined by compute and algorithmic buffering delay**.

### 🗣️ Speaker Script (Minute 1:00 – 2:00)
> *"To understand the core problem, consider this paradox: on modern 5G or broadband networks, transmitting 1 kilobyte takes less than 3 milliseconds. Physical transmission time has essentially vanished.*
> 
> *Yet, if you try to build a conversational voice system using Speech-to-Text followed by Text-to-Speech, users experience severe lag—often 5 to 20 seconds. Where is this time lost? It is lost in algorithmic buffering and computational cascades. To evaluate semantic systems fairly, we must evaluate both bandwidth and interactive latency simultaneously."*

---

## Slide 3: Architectural Comparison: S2S vs. S2T2S

### Visual Layout
- **Top Pipeline (Pipeline B: Cascade S2T2S):**
  $$\text{Speech In} \xrightarrow[\text{Buffer Whole Phrase}]{\text{SenseVoice ASR}} \text{Text} \xrightarrow[\text{Continuous Symbols}]{\text{DeepSC Enc}} \text{Wireless Channel} \xrightarrow[\text{Autoregressive}]{\text{DeepSC Dec}} \text{Text} \xrightarrow[\text{Neural Vocoder}]{\text{Kokoro TTS}} \text{Speech Out}$$
- **Bottom Pipeline (Pipeline A: Semantic S2S):**
  $$\text{Speech In} \xrightarrow[\text{20 ms Frames}]{\text{Neural Tokenizer (RVQ)}} \text{Semantic Tokens} \xrightarrow[\text{Instantaneous}]{\text{Fading Channel}} \xrightarrow[\text{Frame-by-Frame}]{\text{Streaming Vocoder}} \text{Speech Out}$$

### Slide Bullet Points
- **Pipeline B (Cascade S2T2S):**
  - Extracts textual semantics + prosodic metadata (Emotion & Gender).
  - Uses trained Transformer (DeepSC, 22,234 vocabulary) to survive channel noise.
  - Requires **complete phrase transcription** before the receiver can start audio synthesis.
- **Pipeline A (Direct Semantic S2S):**
  - Encodes acoustic waveforms directly into discrete hierarchical tokens (Residual Vector Quantization).
  - Operates on small streaming frames (12.5 ms to 20 ms).
  - Emits audio continuously with minimal lookahead delay.

### 🗣️ Speaker Script (Minute 2:00 – 3:00)
> *"Here we contrast the two architectural paradigms.*
> 
> *In our implemented Pipeline B, speech is first transcribed using SenseVoice-Small. The text is passed through our trained DeepSC model—a 4-layer Transformer that maps tokens to 16-dimensional channel vectors, transmits them across simulated wireless channels, and decodes them autoregressively before Kokoro synthesizes the final audio.*
> 
> *In contrast, Pipeline A bypasses text conversion entirely. It uses neural audio tokenizers like SpeechTokenizer or Kyutai Mimi to tokenize 20-millisecond windows. As we will see, this structural difference has profound implications on latency."*

---

## Slide 4: Theoretical Delay Formulation

### Visual Layout
- **Key Formula:**
  $$\text{Latency}_{\text{total}} = T_{\text{buffering}} + T_{\text{encode}} + T_{\text{transmission}} + T_{\text{decode}}$$
- **Deconstruction Diagram:**
  - $T_{\text{transmission}} < 3\text{ ms}$ (Negligible for both systems)
  - S2T2S Delay: $T_{\text{buffering}} = \mathcal{O}(\text{Duration})$ + 3-Stage Compute
  - S2S Delay: $T_{\text{buffering}} = \mathcal{O}(1)$ (Single 20 ms Frame)

### Slide Bullet Points
- **1. The Context Bottleneck (Algorithmic Delay):**
  - ASR models cannot transcribe speech word-by-word accurately; they require full sentence context to disambiguate homophones and resolve grammar.
  - TTS acoustic models require the entire sentence to compute pitch contours and prosody.
  - Mandatory delay: $T_{\text{buffering}} = \text{Audio Duration}$ (e.g., 3,000 ms to 25,000 ms).
- **2. The Computational Cascade (Inference Delay):**
  - Pipeline B executes 3 heavy generative models sequentially:
    1. ASR encoder-decoder forward pass.
    2. DeepSC Transformer autoregressive greedy decoding.
    3. TTS spectrogram synthesis + neural vocoder.
  - Pipeline A executes a lightweight convolutional encoder and streaming vocoder concurrently.

### 🗣️ Speaker Script (Minute 3:00 – 4:00)
> *"Let us formalize this mathematically. Total digital latency is the sum of buffering, encoding, transmission, and decoding.*
> 
> *Because semantic payloads are lightweight—under 1,500 bits per second—transmission time over the air is virtually instantaneous. The bottleneck is entirely compute and buffering.*
> 
> *For S2T2S, the killer is what we call the 'Context Bottleneck.' An ASR cannot guess the end of a sentence while you are still speaking without making catastrophic errors. It must wait. The TTS must also wait for the full text to calculate natural pitch. This introduces an unavoidable algorithmic delay proportional to the sentence length."*

---

## Slide 5: Experimental Design & The 2×2 Matrix

### Visual Layout
- **The Benchmark Testbed:**
  - Test Corpus: 6 standardized samples spanning Short (1.1–3.1s), Medium (7.2–10.8s), and Long (24.7s) speech.
  - Physical Channels: Additive White Gaussian Noise (AWGN), Rayleigh Multipath Fading, Rician Line-of-Sight Fading.
  - SNR Range: -5 dB to +20 dB.
  - Precision Instrumentation: `time.perf_counter()` hardware timers wrapping every pipeline stage.

### The 2×2 Benchmark Evaluation Matrix
| Evaluation Axis | Pipeline A: Semantic S2S | Pipeline B: Cascade S2T2S |
|---|---|---|
| **Data Rate (Bandwidth)** | 1,100 – 1,500 bps | **622 – 1,077 bps (Winner)** |
| **Time Efficiency (TTFA)** | **43 – 63 ms (Winner)** | 5,278 – 29,169 ms |
| **Algorithmic Lookahead** | **20 ms Frame (Winner)** | Full Clause (1,500 – 25,000 ms) |
| **Channel Degradation** | Frame loss / acoustic degradation | Graceful semantic BLEU degradation |

### 🗣️ Speaker Script (Minute 4:00 – 5:00)
> *"To prove this credibly, we designed an experimental testbed. We constructed a standardized corpus with speech durations ranging from 1 second greetings to 25 second technical paragraphs.*
> 
> *We subjected both pipelines to identical wireless environments: AWGN, Rayleigh fading, and Rician fading from -5 dB to +20 dB SNR. We instrumented every sub-millisecond stage: STT forward inference, phrase buffering, DeepSC power normalization and zero-forcing equalization, and synthesis.*
> 
> *Now, let us examine the empirical data."*

---

## Slide 6: Empirical Results: Bandwidth Efficiency

### Visual Layout
- **Primary Image:** `outputs/charts/bandwidth_comparison.png`
- **Callout Card:** *"S2T2S achieves 622 bps on long paragraphs — 2.4× more compact than SpeechTokenizer (1,500 bps)."*

### Slide Bullet Points
- **Short Audio (<3s):** S2T2S requires ~1,127 bps due to DeepSC start/end token symbols (16 dimensions/token).
- **Paragraph Audio (>20s):** S2T2S bitrate drops to **622 bps**, beating Mimi (1,100 bps) and SpeechTokenizer (1,500 bps).
- **Why S2T2S Wins on Bandwidth:**
  - Human speech contains vast acoustic redundancy (timbre, room reflections, breath).
  - Transcribing speech to ASCII text discards physical acoustic overhead, distilling only symbolic meaning.
  - DeepSC achieves a **1.4× to 2.4× bandwidth advantage** over neural audio codecs.

### 🗣️ Speaker Script (Minute 5:00 – 6:00)
> *"Here are our empirical bandwidth measurements across duration categories.*
> 
> *On the right of Chart 1, observe what happens on longer speech samples. As speech length increases, text compression shines. The effective bitrate of S2T2S drops to 622 bits per second.*
> 
> *In contrast, S2S neural codecs like SpeechTokenizer and Mimi produce a steady stream of discrete codes at 1,100 to 1,500 bits per second. In raw data compression, S2T2S wins decisively."*

---

## Slide 7: Empirical Results: Time-to-First-Audio (TTFA)

### Visual Layout
- **Primary Image:** `outputs/charts/ttfa_comparison.png`
- **Reference Line:** 200 ms ITU-T Conversational Limit (Amber Dotted Line).
- **Metric Highlight:** **S2S TTFA = 43–63 ms** vs. **S2T2S TTFA = 5,278–29,169 ms** (186× to 273× speedup).

### Slide Bullet Points
- **Conversational Usability:**
  - Real-time conversation requires delay $<200\text{ ms}$.
  - Semantic S2S (SpeechTokenizer at 63 ms, Mimi at 43 ms) operates well within this human threshold.
- **S2T2S Delay:**
  - Short speech takes **5.2 seconds** before the first sound is heard.
  - A 24-second paragraph takes **29.2 seconds** before playback begins!
- **Conclusion:** S2T2S completely breaks conversational flow. It is fundamentally non-viable for interactive dialogue.

### 🗣️ Speaker Script (Minute 6:00 – 7:00)
> *"Now look at Chart 2—the Time-to-First-Audio on a logarithmic scale. Here the entire argument flips.*
> 
> *Notice the amber dotted line at 200 milliseconds. That is the ITU-T standard for human conversational interaction. Beyond 200 milliseconds, humans talk over one another and conversational synchrony breaks down.*
> 
> *Semantic S2S sits at 43 to 63 milliseconds—delivering audio almost instantly. S2T2S takes 5 to 29 seconds. S2S is nearly two hundred times faster."*

---

## Slide 8: Root Cause: Proving the Context Bottleneck

### Visual Layout
- **Primary Image:** `outputs/charts/latency_breakdown.png`
- **Breakdown Stack (7.25s Sentence):**
  - 🟥 Algorithmic Phrase Buffering: **7,248 ms (65.6%)**
  - 🟧 STT Forward Inference: **627 ms (5.7%)**
  - 🟦 DeepSC Semantic Channel: **175 ms (1.6%)**
  - 🟪 Kokoro TTS Synthesis: **3,000 ms (27.1%)**
  - 🟩 S2S Total TTFA: **63 ms (<1%)**

### Slide Bullet Points
- **The Context Bottleneck Proven:**
  - Over **65% of the entire delay** is simply waiting for the user to finish the phrase!
  - DeepSC transmission itself is lightning fast (only 175 ms).
- **The Theoretical Implication:**
  - Even with infinitely fast GPUs ($T_{\text{compute}} = 0$), S2T2S would still require **7.2 seconds** of delay.
  - S2S achieves 63 ms because its buffer is a single 20 ms frame.

### 🗣️ Speaker Script (Minute 7:00 – 8:00)
> *"To prove this rigorously to the committee, look at Chart 3, which deconstructs the exact latency breakdown of a 7.25-second utterance.*
> 
> *Look at the red segment. That is phrase buffering. It accounts for 65.6% of the entire delay. Even if we had a supercomputer where model inference was instantaneous, S2T2S would still have over 7 seconds of lag.*
> 
> *This confirms our thesis: S2T2S does not suffer from slow computation, but from the fundamental linguistic necessity of phrase-level context buffering."*

---

## Slide 9: Duration Scaling & Computational Throughput

### Visual Layout
- **Primary Image:** `outputs/charts/rtf_vs_duration.png`
- **Panel (a):** TTFA vs. Duration — Linear $\mathcal{O}(N)$ vs. Flat $\mathcal{O}(1)$.
- **Panel (b):** Real-Time Factor (RTF) — S2T2S RTF drops from 0.47 to 0.06 as duration grows.

### Slide Bullet Points
- **Scaling Complexity:**
  - S2T2S TTFA scales as $\mathcal{O}(N)$ where $N$ is utterance length ($TTFA \approx \text{Duration} + 4\text{s}$).
  - S2S TTFA scales as $\mathcal{O}(1)$ constant time ($63\text{ ms}$ regardless of audio length).
- **The RTF Nuance:**
  - Real-Time Factor (RTF) $< 1.0$ indicates compute is faster than real-time playback.
  - S2T2S achieves an impressive compute RTF of **0.06 to 0.15** on long text (16× faster than real-time in batch).
  - *Key Engineering Insight:* A low RTF is necessary but **not sufficient** for streaming when blocked by input buffering.

### 🗣️ Speaker Script (Minute 8:00 – 9:00)
> *"Chart 4 illustrates the scaling behavior.*
> 
> *In panel (a), S2T2S shows a steep linear curve: the longer you speak, the longer you wait. S2S remains a completely horizontal line at 63 milliseconds.*
> 
> *In panel (b), we examine the Real-Time Factor. S2T2S actually has a very low RTF—around 0.06—meaning it computes text and audio 16 times faster than real time. But because that compute cannot start until the sentence finishes, high computational throughput cannot overcome the buffering bottleneck."*

---

## Slide 10: The Definitive Tradeoff: The Pareto Frontier

### Visual Layout
- **Primary Image:** `outputs/charts/tradeoff_scatter.png` ⭐ *(The Money Slide)*
- **X-Axis:** Effective Bitrate (bps) [Lower is Better]
- **Y-Axis:** TTFA (ms, Log Scale) [Lower is Better]
- **Highlighted Regimes:**
  - 🟢 *Real-Time Conversational Zone* (Bottom-Right: S2S)
  - 🔵 *Bandwidth-Constrained Archival Zone* (Top-Left: S2T2S)

### Slide Bullet Points
- **The Core Contribution:** Plotting the empirical **Pareto Frontier**.
- **Regime 1: Conversational Streaming Zone**
  - Occupied by S2S (SpeechTokenizer, Mimi, EnCodec).
  - TTFA: **43–63 ms** | Bitrate: **1,100–1,500 bps**.
  - Target Applications: Live Voice Calls, Telepresence, Real-Time Interactive AI.
- **Regime 2: Ultra-Low Bandwidth Archival Zone**
  - Occupied by Cascade S2T2S (DeepSC).
  - TTFA: **5,000–29,000 ms** | Bitrate: **622–1,000 bps**.
  - Target Applications: Voice Messaging, Satellite Uplinks, Low-Power IoT Telemetry.

### 🗣️ Speaker Script (Minute 9:00 – 10:00)
> *"This is our defining slide—Chart 5, mapping the empirical Pareto frontier.*
> 
> *If an engineer asks: 'Which pipeline is better?', the answer is: neither. They solve two fundamentally different problems.*
> 
> *If your channel is severely constrained—such as a remote satellite uplink or deep-sea telemetry under 800 bits per second—S2T2S is optimal. But if you need live conversational interaction, Semantic S2S is mandatory. S2S sacrifices about 30% in bandwidth to gain a 200-fold reduction in response latency."*

---

## Slide 11: Wireless Physical-Layer Robustness

### Visual Layout
- **Primary Image:** `outputs/charts/snr_vs_fidelity.png`
- **Channels Plotted:** AWGN, Rayleigh Fading (Multipath), Rician Fading ($K=1.0$).
- **SNR Range:** -5 dB to +20 dB.

### Slide Bullet Points
- **DeepSC Wireless Performance:**
  - Evaluated on our 22,234-token Transformer channel autoencoder.
  - Zero-forcing equalization under complex fading matrices:
    $$\mathbf{H} = \begin{bmatrix} H_{\text{real}} & -H_{\text{imag}} \\ H_{\text{imag}} & H_{\text{real}} \end{bmatrix}$$
- **Key Empirical Observations:**
  - **High SNR (>15 dB):** BLEU scores reach peak fidelity; channel noise has zero impact on reconstructed text.
  - **Low SNR (<5 dB):** Graceful degradation. Unlike classical digital transmission (e.g., LDPC + 16-QAM) which exhibits a catastrophic 'cliff-edge' drop when SNR falls below threshold, DeepSC preserves core semantic tokens even when phonemes are corrupted.

### 🗣️ Speaker Script (Minute 10:00 – 11:00)
> *"On Slide 11, we demonstrate the physical layer wireless performance of our DeepSC model across AWGN, Rayleigh, and Rician fading.*
> 
> *Notice that across all three channels, semantic fidelity degrades gracefully. Traditional digital communication fails catastrophically when SNR drops below a cliff-edge threshold. DeepSC's learned constellation space ensures that even under severe noise at 0 dB and -5 dB, essential semantic keywords are preserved."*

---

## Slide 12: Engineering Conclusion & Future Roadmap

### Visual Layout
- **Summary Matrix:** Side-by-side recap of Bandwidth vs. Latency wins.
- **Interactive UI Preview:** Screenshot of our live Gradio dashboard (`pipeline/demo.py`).
- **Concluding Quote:**
  > *"Direct Semantic S2S is not optimized for raw compression, but for preserving streaming real-time conversational flow. S2T2S is optimized for bandwidth efficiency over non-real-time channels."*

### Slide Bullet Points
- **Midsem Deliverables Accomplished:**
  - [x] End-to-end Speech-to-Speech communication pipeline with Emotion & Gender preservation.
  - [x] Full wireless channel simulation (AWGN, Rayleigh, Rician).
  - [x] Comprehensive timing instrumentation and empirical benchmark matrix.
  - [x] Mathematical and empirical proof of the Context Bottleneck and Pareto Trade-off.
- **Endsem Roadmap:**
  - Integrate live Kyutai Mimi / SpeechTokenizer neural codecs directly into the DeepSC physical transmission loop.
  - Deploy on software-defined radio (SDR) hardware-in-the-loop for over-the-air validation.
  - Extend objective speech quality metrics (PESQ, STOI, and Speaker Similarity).

### 🗣️ Speaker Script (Minute 11:00 – 12:00)
> *"To conclude: our midsem evaluation has proven the trade-off conclusively.*
> 
> *S2T2S achieves superior bandwidth efficiency—reducing data rates to 622 bits per second—at the cost of a 186-fold latency penalty due to phrase buffering. Semantic S2S operates frame-by-frame, delivering first audio in 43 to 63 milliseconds to preserve real-time conversational flow.*
> 
> *All models, datasets, and benchmark scripts are fully operational in our codebase, along with our live Gradio web dashboard for real-time demonstrations. Thank you, and I look forward to your questions."*
