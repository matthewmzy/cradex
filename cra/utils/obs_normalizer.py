"""Running observation normalizer for RL training stability.

Maintains per-feature running mean and variance using Welford's
online algorithm.  Normalizes observations to approximately zero
mean and unit variance, then clips to [-clip_range, clip_range].
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ObsNormalizer(nn.Module):
    """Online observation normalizer with running statistics.

    Usage
    -----
    >>> norm = ObsNormalizer(obs_dim=118)
    >>> obs_normed = norm(obs)        # update stats + normalize
    >>> obs_normed = norm(obs, update=False)  # normalize only (eval)
    """

    def __init__(
        self,
        obs_dim: int,
        clip_range: float = 5.0,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.clip_range = clip_range
        self.epsilon = epsilon

        # Running statistics (not trainable)
        self.register_buffer("mean", torch.zeros(obs_dim))
        self.register_buffer("var", torch.ones(obs_dim))
        self.register_buffer("count", torch.tensor(0.0))

    @torch.no_grad()
    def update(self, obs: torch.Tensor) -> None:
        """Update running statistics with a batch of observations.

        Parameters
        ----------
        obs : (batch, obs_dim)
        """
        batch_mean = obs.mean(dim=0)
        batch_var = obs.var(dim=0, unbiased=False)
        batch_count = obs.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count.clamp(min=1)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count.clamp(min=1)
        new_var = m2 / total_count.clamp(min=1)

        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(total_count)

    def normalize(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalize observations using current running stats."""
        normed = (obs - self.mean) / (self.var.sqrt() + self.epsilon)
        return normed.clamp(-self.clip_range, self.clip_range)

    def forward(
        self, obs: torch.Tensor, update: bool = True
    ) -> torch.Tensor:
        """Normalize observations, optionally updating running stats.

        Parameters
        ----------
        obs    : (batch, obs_dim)
        update : whether to update running statistics (set False for eval)
        """
        if update and self.training:
            self.update(obs)
        return self.normalize(obs)
