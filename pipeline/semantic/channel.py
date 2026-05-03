"""
Wireless channel simulation for semantic communication.

Supports:
 - AWGN  (Additive White Gaussian Noise)
 - Rayleigh fading
 - Rician fading (K-factor = 1, matching training code)

Two forward modes:
 - forward()                — simple noise addition (for mock mode)
 - forward_training_style() — exact channel from training (2×2 matrix fading)
"""

import math
import torch
import numpy as np


class ChannelSimulator:
    """Simulate the effect of a noisy wireless channel on transmitted symbols."""

    def __init__(self, channel_type: str = "AWGN", snr_db: float = 10.0):
        self.channel_type = channel_type.upper()
        self.snr_db = snr_db

    # ------------------------------------------------------------------ #
    #  Simple forward (for MOCK mode)                                     #
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass tensor *x* through the simulated channel.

        Parameters
        ----------
        x : torch.Tensor
            The semantic feature vector (transmitted signal).

        Returns
        -------
        torch.Tensor
            Received signal after channel effects + noise.
        """
        signal_power = x.pow(2).mean().item()
        noise_std = self._noise_std(signal_power)

        if self.channel_type == "AWGN":
            return self._awgn(x, noise_std)
        elif self.channel_type == "RAYLEIGH":
            return self._rayleigh(x, noise_std)
        elif self.channel_type == "RICIAN":
            return self._rician(x, noise_std)
        else:
            raise ValueError(f"Unknown channel type: {self.channel_type}")

    # ------------------------------------------------------------------ #
    #  Training-style forward (for REAL mode — matches training code)     #
    # ------------------------------------------------------------------ #

    def forward_training_style(self, x: torch.Tensor, n_var: float) -> torch.Tensor:
        """
        Pass signal through channel using the EXACT same implementation
        as the training code (from DeepSC/utils.py Channels class).

        This is critical — the model was trained with this specific noise
        distribution, so inference must use the same one.

        Parameters
        ----------
        x : torch.Tensor
            Power-normalized transmitted signal from channel encoder.
        n_var : float
            Noise standard deviation (from SNR_to_noise).
        """
        device = x.device

        if self.channel_type == "AWGN":
            return self._training_awgn(x, n_var, device)
        elif self.channel_type == "RAYLEIGH":
            return self._training_rayleigh(x, n_var, device)
        elif self.channel_type == "RICIAN":
            return self._training_rician(x, n_var, device)
        else:
            raise ValueError(f"Unknown channel type: {self.channel_type}")

    # ------------------------------------------------------------------ #
    #  Training-style channel implementations (from DeepSC utils.py)      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _training_awgn(x: torch.Tensor, n_var: float, device) -> torch.Tensor:
        """AWGN channel — exactly as in training."""
        return x + torch.normal(0, n_var, size=x.shape).to(device)

    @staticmethod
    def _training_rayleigh(x: torch.Tensor, n_var: float, device) -> torch.Tensor:
        """Rayleigh fading — exactly as in training (2×2 complex channel matrix)."""
        shape = x.shape
        H_real = torch.normal(0, math.sqrt(1 / 2), size=[1]).to(device)
        H_imag = torch.normal(0, math.sqrt(1 / 2), size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(x.view(shape[0], -1, 2), H)
        Rx_sig = Tx_sig + torch.normal(0, n_var, size=Tx_sig.shape).to(device)
        # Channel estimation (perfect CSI)
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)
        return Rx_sig

    @staticmethod
    def _training_rician(x: torch.Tensor, n_var: float, device, K: float = 1.0) -> torch.Tensor:
        """Rician fading — exactly as in training (K=1)."""
        shape = x.shape
        mean = math.sqrt(K / (K + 1))
        std = math.sqrt(1 / (K + 1))
        H_real = torch.normal(mean, std, size=[1]).to(device)
        H_imag = torch.normal(mean, std, size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(x.view(shape[0], -1, 2), H)
        Rx_sig = Tx_sig + torch.normal(0, n_var, size=Tx_sig.shape).to(device)
        # Channel estimation (perfect CSI)
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)
        return Rx_sig

    # ------------------------------------------------------------------ #
    #  Simple channel implementations (for MOCK mode)                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _awgn(x: torch.Tensor, noise_std: float) -> torch.Tensor:
        """Additive White Gaussian Noise."""
        return x + torch.randn_like(x) * noise_std

    @staticmethod
    def _rayleigh(x: torch.Tensor, noise_std: float) -> torch.Tensor:
        """Rayleigh fading channel (no line-of-sight component)."""
        h_real = torch.randn_like(x) / np.sqrt(2)
        h_imag = torch.randn_like(x) / np.sqrt(2)
        h_mag = (h_real ** 2 + h_imag ** 2).sqrt()
        return h_mag * x + torch.randn_like(x) * noise_std

    @staticmethod
    def _rician(x: torch.Tensor, noise_std: float, K: float = 2.0) -> torch.Tensor:
        """Rician fading channel (with line-of-sight, K-factor = 2)."""
        nu = np.sqrt(K / (K + 1))
        sigma = np.sqrt(1 / (2 * (K + 1)))
        h_real = nu + sigma * torch.randn_like(x)
        h_imag = sigma * torch.randn_like(x)
        h_mag = (h_real ** 2 + h_imag ** 2).sqrt()
        return h_mag * x + torch.randn_like(x) * noise_std

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _noise_std(self, signal_power: float) -> float:
        snr_linear = 10 ** (self.snr_db / 10)
        return np.sqrt(signal_power / snr_linear) if snr_linear > 0 else 0.0

    def __repr__(self) -> str:
        return f"ChannelSimulator(type={self.channel_type}, SNR={self.snr_db} dB)"
