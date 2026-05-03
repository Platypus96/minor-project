# 🧠 DeepSC Training on Google Colab — Step-by-Step Guide

Follow these steps **exactly** in Google Colab to train the DeepSC model.

> **Time:** ~10 min setup + 3-5 hours unattended training.
> **GPU:** Free T4 (16 GB VRAM) — more than enough.

---

## Step 0: Open Colab & Enable GPU

1. Go to [https://colab.research.google.com](https://colab.research.google.com)
2. Click **"New Notebook"**
3. Go to **Runtime → Change runtime type → GPU → T4 → Save**

---

## Step 1: Setup Environment

**Create a new code cell and paste:**

```python
# Cell 1 — Setup
!git clone https://github.com/13274086/DeepSC.git
%cd DeepSC

# Install dependencies
!pip install torch torchvision torchaudio
!pip install nltk w3lib

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

# Verify GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

---

## Step 2: Fix Hardcoded Paths in Repository

The DeepSC repository contains hardcoded absolute paths to the author's original server (`/import/antennas/...`). Run this cell to automatically fix them to point to your local Colab folders:

**New cell:**
```python
# Cell 2 — Fix hardcoded paths
!sed -i 's|/import/antennas/Datasets/hx301/europarl/en|./txt/en|g' dataset/preprocess_text.py
!sed -i 's|/import/antennas/Datasets/hx301/europarl/en|./txt/en|g' parameters.py
!sed -i 's|/import/antennas/Datasets/hx301/europarl/vocab.json|./data/vocab.json|g' parameters.py
!sed -i 's|/import/antennas/Datasets/hx301/europarl/train_data.npy|./data/train_data.npy|g' parameters.py
!sed -i 's|/import/antennas/Datasets/hx301/europarl/test_data.npy|./data/test_data.npy|g' parameters.py
```

---

## Step 3: Download & Preprocess the Europarl Dataset

**New cell:**

```python
# Cell 3 — Dataset
!mkdir -p data
!wget -q http://www.statmt.org/europarl/v7/europarl.tgz
!tar zxf europarl.tgz

# Preprocess (creates training/test splits)
!python dataset/preprocess_text.py \
    --input-data-dir=./txt/en \
    --output-train-data=./data/train_data.npy \
    --output-test-data=./data/test_data.npy \
    --output-vocab=./data/vocab.json
```

> ⏱ This takes ~5 min for download + preprocessing.

---

## Step 4: Train DeepSC — AWGN Channel

**New cell:**

```python
# Cell 4 — Train on AWGN channel
!python main.py \
    --bs=64 \
    --train-snr=6 \
    --channel=AWGN \
    --train-with-mine \
    --vocab-file=./data/vocab.json \
    --checkpoint-path=./checkpoints/awgn
```

> ⏱ This runs for ~2-3 hours on a T4. **You can close your laptop** — Colab keeps running.
>
> ⚠ **Important:** Keep the browser tab open (or use Colab's "Prevent disconnection" trick: open browser console with F12 and paste `setInterval(() => document.querySelector("colab-connect-button").click(), 60000)`)

---

## Step 5 (Optional): Train on Rayleigh & Rician Channels

**New cell:**

```python
# Cell 5a — Rayleigh
!python main.py \
    --bs=64 \
    --train-snr=6 \
    --channel=Rayleigh \
    --train-with-mine \
    --vocab-file=./data/vocab.json \
    --checkpoint-path=./checkpoints/rayleigh
```

```python
# Cell 5b — Rician
!python main.py \
    --bs=64 \
    --train-snr=6 \
    --channel=Rician \
    --train-with-mine \
    --vocab-file=./data/vocab.json \
    --checkpoint-path=./checkpoints/rician
```

---

## Step 6: Save Checkpoints to Google Drive

**New cell:**

```python
# Cell 6 — Save to Google Drive
from google.colab import drive
drive.mount('/content/drive')

import shutil
shutil.copytree('./checkpoints', '/content/drive/MyDrive/deepsc_checkpoints', dirs_exist_ok=True)
print("✅ Checkpoints saved to Google Drive!")
```

---

## Step 7: Download to Your Local Machine

**Option A — From Google Drive:**
1. Go to [drive.google.com](https://drive.google.com)
2. Find `deepsc_checkpoints/` folder
3. Right-click → Download

**Option B — Direct download from Colab:**
```python
# Cell 7 — Direct download
!zip -r checkpoints.zip ./checkpoints/
from google.colab import files
files.download('checkpoints.zip')
```

---

## Step 8: Place Checkpoints Locally

After downloading, extract the checkpoints and place them:

```
minor project/
  checkpoints/
    awgn/
      best.pth        ← (or whatever the file is named)
    rayleigh/
      best.pth
    rician/
      best.pth
```

Then update your `.env` file:
```
CHECKPOINT_PATH=./checkpoints/awgn/best.pth
```

The pipeline will automatically detect and load the real model!

---

## Troubleshooting

| Issue | Fix |
|---|---|
| Colab disconnects | Use the anti-disconnect JS trick above, or use **Kaggle** instead (30 free hrs/week) |
| Out of memory | Reduce batch size: `--bs=32` |
| Dataset download fails | Try a mirror: `!wget http://www.statmt.org/europarl/v7/de-en.tgz` |
| `nltk` errors | Run `nltk.download('punkt')` and `nltk.download('punkt_tab')` |
| Training is slow | Make sure GPU runtime is selected (Runtime → Change runtime type) |

---

## Quick Evaluation (Optional)

After training, test the model quality:

```python
# Cell 7 — Evaluate
!python evaluation.py \
    --checkpoint-path=./checkpoints/awgn \
    --channel=AWGN
```

This prints BLEU scores at different SNR levels — you'll want BLEU > 0.8 at SNR 10 dB.
