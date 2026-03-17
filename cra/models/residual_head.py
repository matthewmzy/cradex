"""Residual policy head: produces action corrections conditioned on
the current observation and an adaptation latent vector.

Each CRA stage has its own ResidualHead. The residual is additive:
    a_total = pi_0(o) + sum_i rho_i(o, z_i)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualHead(nn.Module):
    """MLP that outputs an action residual given (observation, latent).

    Architecture
    ------------
    Input:  concatenation of current observation and adaptation latent
    MLP:    configurable hidden layers with ELU activation
    Output: delta_action ∈ R^action_dim (unbounded; clipping at action level)

    The output is initialized near zero so that adding a new stage
    does not immediately disrupt the existing composite policy.
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        action_dim: int,
        hidden_dims: list[int] | None = None,
        activation: type[nn.Module] = nn.ELU,
        init_scale: float = 0.01,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        input_dim = obs_dim + latent_dim
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(activation())
            prev = h
        layers.append(nn.Linear(prev, action_dim))
        self.mlp = nn.Sequential(*layers)

        self._init_weights(init_scale)

    def _init_weights(self, scale: float) -> None:
        """Initialize output layer near zero for smooth stage addition."""
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.zeros_(layer.bias)
        # Last linear layer: very small weights so initial residual ≈ 0
        last_linear = self.mlp[-1]
        assert isinstance(last_linear, nn.Linear)
        nn.init.uniform_(last_linear.weight, -scale, scale)
        nn.init.zeros_(last_linear.bias)

    def forward(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Compute action residual.

        Parameters
        ----------
        obs : (batch, obs_dim)
        z   : (batch, latent_dim)

        Returns
        -------
        delta_action : (batch, action_dim)
        """
        x = torch.cat([obs, z], dim=-1)
        return self.mlp(x)
