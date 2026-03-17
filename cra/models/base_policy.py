"""Base actor-critic MLP policy for dexterous manipulation.

This serves as the Stage-0 (base) policy in the CRA pipeline.
It is trained with fixed or narrow-range domain randomization
on nominal environment parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: list[int],
    activation: type[nn.Module] = nn.ELU,
    output_activation: type[nn.Module] | None = None,
) -> nn.Sequential:
    """Construct an MLP with the given layer sizes."""
    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(activation())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Standard MLP actor-critic with diagonal Gaussian policy.

    Architecture
    ------------
    Actor:  obs -> MLP -> action_mean
            log_std is a learnable parameter vector (state-independent).
    Critic: obs -> MLP -> scalar value
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        actor_hidden_dims: list[int] | None = None,
        critic_hidden_dims: list[int] | None = None,
        activation: type[nn.Module] = nn.ELU,
        init_noise_std: float = 1.0,
        fixed_std: bool = False,
    ) -> None:
        super().__init__()
        if actor_hidden_dims is None:
            actor_hidden_dims = [512, 256, 128]
        if critic_hidden_dims is None:
            critic_hidden_dims = [512, 256, 128]

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Actor network: obs -> action mean
        self.actor = _build_mlp(obs_dim, action_dim, actor_hidden_dims, activation)

        # Learnable log standard deviation
        self.log_std = nn.Parameter(
            torch.ones(action_dim) * torch.tensor(init_noise_std).log()
        )
        self.fixed_std = fixed_std

        # Critic network: obs -> value
        self.critic = _build_mlp(obs_dim, 1, critic_hidden_dims, activation)

        # Weight initialization
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply orthogonal initialization (standard for PPO)."""
        for module in [self.actor, self.critic]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=1.0)
                    nn.init.zeros_(layer.bias)
        # Smaller gain for the output layers
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass.

        Returns
        -------
        action_mean : (batch, action_dim)
        action_std  : (batch, action_dim)  (broadcast from learnable log_std)
        value       : (batch, 1)
        """
        action_mean = self.actor(obs)
        action_std = self.log_std.exp().expand_as(action_mean)
        value = self.critic(obs)
        return action_mean, action_std, value

    def get_action_mean(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic forward through actor only."""
        return self.actor(obs)

    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action and return (action, log_prob, value).

        Parameters
        ----------
        obs : (batch, obs_dim)
        deterministic : if True, return the mean action.

        Returns
        -------
        action   : (batch, action_dim)
        log_prob : (batch,)
        value    : (batch, 1)
        """
        action_mean, action_std, value = self.forward(obs)
        dist = Normal(action_mean, action_std)
        if deterministic:
            action = action_mean
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate given actions under current policy.

        Used by PPO to compute the ratio pi(a|s) / pi_old(a|s).

        Returns
        -------
        log_prob : (batch,)
        entropy  : (batch,)
        value    : (batch, 1)
        """
        action_mean, action_std, value = self.forward(obs)
        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value
