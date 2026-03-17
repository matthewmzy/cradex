"""Adaptation encoder: GRU-based module that maps observation-action
history to a compact latent vector for online system identification.

Each CRA stage has its own AdaptationEncoder that is forced to learn
to identify ONE specific axis of domain variation (e.g. gravity,
friction, mass) from the recent interaction history.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AdaptationEncoder(nn.Module):
    """GRU encoder that produces a latent context vector from history.

    Architecture
    ------------
    Input:  (obs_history, action_history) of shape (B, W, obs_dim+act_dim)
    GRU:    2-layer GRU with hidden_dim units
    Output: linear projection of final hidden state -> z ∈ R^latent_dim
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 16,
        num_layers: int = 2,
        window_size: int = 50,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.input_dim = obs_dim + action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.window_size = window_size

        # Input normalization (running stats updated during training)
        self.input_norm = nn.LayerNorm(self.input_dim)

        # GRU processes the temporal sequence
        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # Project final hidden state to latent
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        for layer in self.projector:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.1)
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        obs_history: torch.Tensor,
        action_history: torch.Tensor,
    ) -> torch.Tensor:
        """Encode observation-action history into latent context.

        Parameters
        ----------
        obs_history    : (batch, window, obs_dim)
        action_history : (batch, window, action_dim)

        Returns
        -------
        z : (batch, latent_dim)  — the adaptation latent for this axis.
        """
        # Concatenate obs and action along feature dimension
        x = torch.cat([obs_history, action_history], dim=-1)  # (B, W, input_dim)
        x = self.input_norm(x)

        # Run through GRU
        _, h_n = self.gru(x)  # h_n: (num_layers, B, hidden_dim)

        # Use the last layer's hidden state
        h_last = h_n[-1]  # (B, hidden_dim)

        # Project to latent space
        z = self.projector(h_last)  # (B, latent_dim)
        return z


class Conv1DAdaptationEncoder(nn.Module):
    """Alternative 1D-CNN encoder for faster training.

    Can be used as a drop-in replacement for the GRU encoder.
    Sometimes converges faster for shorter windows.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 16,
        window_size: int = 50,
        channels: list[int] | None = None,
    ) -> None:
        super().__init__()
        if channels is None:
            channels = [64, 128, 64]

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.input_dim = obs_dim + action_dim
        self.latent_dim = latent_dim
        self.window_size = window_size

        # Build 1D conv layers
        layers: list[nn.Module] = []
        in_ch = self.input_dim
        for out_ch in channels:
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=5, stride=1, padding=2),
                nn.ELU(),
            ])
            in_ch = out_ch
        self.convs = nn.Sequential(*layers)

        # Global average pooling + projection
        self.projector = nn.Sequential(
            nn.Linear(channels[-1], latent_dim),
        )

    def forward(
        self,
        obs_history: torch.Tensor,
        action_history: torch.Tensor,
    ) -> torch.Tensor:
        """Encode history using 1D convolutions.

        Parameters
        ----------
        obs_history    : (batch, window, obs_dim)
        action_history : (batch, window, action_dim)

        Returns
        -------
        z : (batch, latent_dim)
        """
        x = torch.cat([obs_history, action_history], dim=-1)  # (B, W, C_in)
        x = x.transpose(1, 2)  # (B, C_in, W) for Conv1d
        x = self.convs(x)  # (B, C_out, W)
        x = x.mean(dim=2)  # (B, C_out) global average pool
        z = self.projector(x)  # (B, latent_dim)
        return z
