"""
DeepSC Semantic Encoder/Decoder wrapper.

Two operating modes:
  1. MOCK  — pass-through (text in == text out) for testing without checkpoints.
  2. REAL  — loads trained DeepSC model from a checkpoint file.

Switch to REAL mode once you download the trained checkpoints from Google Colab.
"""

import os
import re
import sys
import json
import math
import unicodedata
import torch
import numpy as np
from .channel import ChannelSimulator
from .deepsc_model import DeepSC, power_normalize, subsequent_mask


# ------------------------------------------------------------------ #
#  Text preprocessing — must match training preprocessing exactly      #
# ------------------------------------------------------------------ #

def _unicode_to_ascii(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def _normalize_string(s: str) -> str:
    """Normalize text exactly as the training preprocessing does."""
    s = _unicode_to_ascii(s)
    s = re.sub(r"([!.?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s)
    s = s.lower().strip()
    return s


def _tokenize(s: str, add_start=True, add_end=True) -> list[str]:
    """Tokenize matching the training script's tokenize() function."""
    # The training script keeps ';' and ',' as separate tokens
    # and removes '?' and '.'
    for p in [";", ","]:
        s = s.replace(p, f" {p}")
    for p in ["?", "."]:
        s = s.replace(p, "")

    tokens = s.split()
    if add_start:
        tokens.insert(0, "<START>")
    if add_end:
        tokens.append("<END>")
    return tokens


class DeepSCWrapper:
    """
    Wraps the DeepSC encoder → channel → decoder pipeline.

    In mock mode the text is converted to a simple numerical embedding,
    passed through the channel, and reconstructed via argmax — just to
    prove the pipeline works end-to-end.  With real checkpoints the
    trained Transformer does the heavy lifting.
    """

    # Default model hyperparameters (must match training config)
    NUM_LAYERS = 4
    D_MODEL = 128
    NUM_HEADS = 8
    DFF = 512
    MAX_LEN = 30
    DROPOUT = 0.1

    def __init__(
        self,
        checkpoint_dir: str | None = None,
        checkpoint_path: str | None = None,  # legacy single-file path
        vocab_path: str | None = None,
        channel_type: str = "AWGN",
        snr_db: float = 10.0,
        device: str = "cpu",
    ):
        self.device = device
        self.channel = ChannelSimulator(channel_type=channel_type, snr_db=snr_db)
        self.mock_mode = True
        self.models = {}  # channel_type -> loaded DeepSC model
        self.token_to_idx = None
        self.idx_to_token = None
        self.pad_idx = 0
        self.start_idx = 1
        self.end_idx = 2
        self.unk_idx = 3
        self.num_vocab = 0

        # --- Resolve checkpoint directory ---
        if checkpoint_dir and os.path.isdir(checkpoint_dir):
            self._checkpoint_dir = checkpoint_dir
        elif checkpoint_path and os.path.isfile(checkpoint_path):
            # Legacy: single checkpoint file → use its parent dir
            self._checkpoint_dir = os.path.dirname(os.path.dirname(checkpoint_path))
        else:
            self._checkpoint_dir = None

        # --- Load vocabulary ---
        vocab_loaded = False
        if vocab_path and os.path.isfile(vocab_path):
            vocab_loaded = self._load_vocab(vocab_path)
        elif self._checkpoint_dir:
            # Look for vocab.json in the checkpoint directory
            candidate = os.path.join(self._checkpoint_dir, "vocab.json")
            if os.path.isfile(candidate):
                vocab_loaded = self._load_vocab(candidate)

        if not vocab_loaded:
            print("[DeepSC] ⚠ No valid vocab.json found — running in MOCK mode.")
            print("[DeepSC]   Download vocab.json from Colab (DeepSC/data/vocab.json)")
            return

        # --- Load models for available channels ---
        if self._checkpoint_dir:
            self._load_channel_models()

        if not self.models:
            print("[DeepSC] ⚠ No checkpoint files loaded — running in MOCK mode.")
            print("[DeepSC]   Place checkpoint .pth files in checkpoints/<channel>/ dirs.")

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def transmit(self, text: str) -> str:
        """
        Full semantic communication pass:
        text → encode → channel → decode → reconstructed text.
        """
        if self.mock_mode:
            return self._mock_transmit(text)
        else:
            return self._real_transmit(text)

    # ------------------------------------------------------------------ #
    #  MOCK mode — word-level simulation                                  #
    # ------------------------------------------------------------------ #

    def _mock_transmit(self, text: str) -> str:
        """
        Simulate semantic comm without a trained model.

        Approach:
         1. Map each word → integer index (one-hot-ish embedding).
         2. Convert to float tensor.
         3. Pass through channel (adds real AWGN / fading noise).
         4. Round back to nearest integer → look up word.

        At high SNR the text survives perfectly.
        At low SNR some words may get corrupted — this is *realistic*
        and shows why DeepSC matters.
        """
        words = text.split()
        if not words:
            return text

        # Build a tiny vocabulary from the input sentence
        vocab = list(dict.fromkeys(words))  # preserve order, unique
        word2idx = {w: i for i, w in enumerate(vocab)}
        idx2word = {i: w for i, w in enumerate(vocab)}

        # Encode: word indices as float tensor (simulate channel symbols)
        indices = torch.tensor(
            [word2idx[w] for w in words], dtype=torch.float32
        ).unsqueeze(0)  # (1, seq_len)

        # Normalize to unit power
        power = indices.pow(2).mean().sqrt().clamp(min=1e-8)
        tx = indices / power

        # Channel
        rx = self.channel.forward(tx)

        # Denormalize
        rx = rx * power

        # Decode: round to nearest valid index
        decoded_indices = rx.squeeze(0).round().long().clamp(0, len(vocab) - 1)
        decoded_words = [idx2word.get(i.item(), "<UNK>") for i in decoded_indices]

        return " ".join(decoded_words)

    # ------------------------------------------------------------------ #
    #  REAL mode — trained DeepSC model                                   #
    # ------------------------------------------------------------------ #

    def _load_vocab(self, vocab_path: str) -> bool:
        """Load vocabulary from JSON file."""
        try:
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab_data = json.load(f)

            # Handle both formats:
            # Format 1 (from preprocess_text.py): {"token_to_idx": {"<PAD>": 0, ...}}
            # Format 2 (flat): {"the": 0, ",": 1, ...}
            if "token_to_idx" in vocab_data:
                self.token_to_idx = vocab_data["token_to_idx"]
            else:
                self.token_to_idx = vocab_data

            self.num_vocab = len(self.token_to_idx)
            self.idx_to_token = {v: k for k, v in self.token_to_idx.items()}

            # Get special token indices
            self.pad_idx = self.token_to_idx.get("<PAD>", 0)
            self.start_idx = self.token_to_idx.get("<START>", 1)
            self.end_idx = self.token_to_idx.get("<END>", 2)
            self.unk_idx = self.token_to_idx.get("<UNK>", 3)

            print(f"[DeepSC] ✅ Loaded vocab: {self.num_vocab} tokens from {vocab_path}")

            # Validate: check if special tokens exist
            if "<PAD>" not in self.token_to_idx:
                print(f"[DeepSC] ⚠ WARNING: vocab.json is missing <PAD>/<START>/<END> tokens!")
                print(f"[DeepSC]   This may be the wrong file. The correct one has ~22,000+ tokens.")
                return False

            return True

        except Exception as e:
            print(f"[DeepSC] ⚠ Failed to load vocab: {e}")
            return False

    def _load_channel_models(self):
        """Load DeepSC models for each channel type found in checkpoint_dir."""
        for channel_name in ["awgn", "rayleigh", "rician"]:
            channel_dir = os.path.join(self._checkpoint_dir, channel_name)
            if not os.path.isdir(channel_dir):
                continue

            # Find the latest checkpoint (highest epoch number)
            pth_files = []
            for fn in os.listdir(channel_dir):
                if fn.endswith(".pth"):
                    pth_files.append(os.path.join(channel_dir, fn))

            if not pth_files:
                continue

            # Sort by filename to get latest epoch
            pth_files.sort()
            latest_ckpt = pth_files[-1]

            try:
                model = DeepSC(
                    num_layers=self.NUM_LAYERS,
                    src_vocab_size=self.num_vocab,
                    trg_vocab_size=self.num_vocab,
                    src_max_len=self.num_vocab,  # training code passes num_vocab as max_len
                    trg_max_len=self.num_vocab,
                    d_model=self.D_MODEL,
                    num_heads=self.NUM_HEADS,
                    dff=self.DFF,
                    dropout=self.DROPOUT,
                ).to(self.device)

                ckpt = torch.load(latest_ckpt, map_location=self.device, weights_only=True)
                model.load_state_dict(ckpt)
                model.eval()

                self.models[channel_name.upper()] = model
                print(f"[DeepSC] ✅ Loaded {channel_name.upper()} model from {latest_ckpt}")

            except Exception as e:
                print(f"[DeepSC] ⚠ Failed to load {channel_name} checkpoint: {e}")

        if self.models:
            self.mock_mode = False
            print(f"[DeepSC] 🚀 REAL mode active — channels: {list(self.models.keys())}")

    def _get_model(self) -> DeepSC:
        """Get the model matching the current channel type, with fallback."""
        channel = self.channel.channel_type.upper()
        if channel in self.models:
            return self.models[channel]
        # Fallback: use any available model
        fallback = next(iter(self.models.values()))
        print(f"[DeepSC] ⚠ No model for {channel}, using fallback")
        return fallback

    def _text_to_indices(self, text: str) -> list[int]:
        """Convert text to token indices using the loaded vocabulary."""
        normalized = _normalize_string(text)
        tokens = _tokenize(normalized)
        indices = []
        for token in tokens:
            if token in self.token_to_idx:
                indices.append(self.token_to_idx[token])
            else:
                indices.append(self.unk_idx)
        return indices

    def _indices_to_text(self, indices: list[int]) -> str:
        """Convert token indices back to text."""
        words = []
        for idx in indices:
            if idx == self.end_idx:
                break
            if idx in (self.pad_idx, self.start_idx):
                continue
            token = self.idx_to_token.get(idx, "<UNK>")
            words.append(token)
        return " ".join(words)

    def _real_transmit(self, text: str) -> str:
        """Transmit text through the real trained DeepSC model."""
        model = self._get_model()
        channel_type = self.channel.channel_type

        # --- Tokenize input ---
        src_indices = self._text_to_indices(text)

        # Truncate to MAX_LEN
        if len(src_indices) > self.MAX_LEN:
            src_indices = src_indices[: self.MAX_LEN]

        src = torch.tensor([src_indices], dtype=torch.long).to(self.device)

        # --- Greedy decode (matches training repo's greedy_decode) ---
        with torch.no_grad():
            result_indices = self._greedy_decode(
                model, src, channel_type,
                max_len=self.MAX_LEN,
            )

        # --- Detokenize ---
        output_indices = result_indices[0].cpu().numpy().tolist()
        reconstructed = self._indices_to_text(output_indices)

        return reconstructed

    def _greedy_decode(
        self,
        model: DeepSC,
        src: torch.Tensor,
        channel_type: str,
        max_len: int = 30,
    ) -> torch.Tensor:
        """
        Greedy autoregressive decoding — matches the training repo's
        greedy_decode() function exactly.
        """
        # Create source mask
        src_mask = (src == self.pad_idx).unsqueeze(-2).type(torch.FloatTensor).to(self.device)

        # Encode
        enc_output = model.encoder(src, src_mask)
        channel_enc_output = model.channel_encoder(enc_output)
        tx_sig = power_normalize(channel_enc_output)

        # Pass through channel (using the SAME channel implementation as training)
        snr_db = self.channel.snr_db
        noise_std = self._snr_to_noise(snr_db)
        rx_sig = self.channel.forward_training_style(tx_sig, noise_std)

        # Channel decode
        memory = model.channel_decoder(rx_sig)

        # Autoregressive decoding
        outputs = torch.ones(src.size(0), 1).fill_(self.start_idx).type_as(src.data)

        for i in range(max_len - 1):
            trg_mask = (outputs == self.pad_idx).unsqueeze(-2).type(torch.FloatTensor)
            look_ahead_mask = subsequent_mask(outputs.size(1)).type(torch.FloatTensor)
            combined_mask = torch.max(trg_mask, look_ahead_mask).to(self.device)

            dec_output = model.decoder(outputs, memory, combined_mask, None)
            pred = model.dense(dec_output)

            # Get the last predicted token
            prob = pred[:, -1:, :]  # (batch, 1, vocab_size)
            _, next_word = torch.max(prob, dim=-1)
            outputs = torch.cat([outputs, next_word], dim=1)

        return outputs

    @staticmethod
    def _snr_to_noise(snr_db: float) -> float:
        """Convert SNR (dB) to noise standard deviation — matches training."""
        snr_linear = 10 ** (snr_db / 10)
        return 1 / np.sqrt(2 * snr_linear)

    # ------------------------------------------------------------------ #
    #  Config                                                              #
    # ------------------------------------------------------------------ #

    def set_channel(self, channel_type: str, snr_db: float):
        """Update channel parameters."""
        self.channel.channel_type = channel_type.upper()
        self.channel.snr_db = snr_db

    def __repr__(self) -> str:
        mode = "MOCK" if self.mock_mode else "REAL"
        channels = list(self.models.keys()) if self.models else []
        return f"DeepSCWrapper(mode={mode}, channels={channels}, {self.channel})"
