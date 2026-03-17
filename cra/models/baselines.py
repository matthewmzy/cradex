"""Baseline policies for comparison with CRA.

- FullDRPolicy      : same architecture as the base ActorCritic, trained with
                      full domain randomization from scratch.
- CurriculumDRPolicy: same architecture as FullDRPolicy, but trained with the
                      same staged DR curriculum as CRA (no frozen cascade).
- RMAPolicy         : Rapid Motor Adaptation style — single adaptation encoder
                      that identifies ALL environment parameters at once.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from cra.models.base_policy import ActorCritic, _build_mlp
from cra.models.adaptation_encoder import AdaptationEncoder


# ======================================================================
# FullDRPolicy — identical architecture to base, different training
# ======================================================================

class FullDRPolicy(ActorCritic):
    """Actor-Critic trained with simultaneous full domain randomization.

    Architecturally identical to the CRA base policy; difference is
    purely in the training regime (all DR axes randomized from step 1).
    This class exists mainly for clarity in experiment configs.
    """
    pass


# ======================================================================
# CurriculumDRPolicy — same architecture, staged DR curriculum
# ======================================================================

class CurriculumDRPolicy(ActorCritic):
    """Actor-Critic trained with the same staged DR curriculum as CRA.

    Architecturally identical to FullDRPolicy / ActorCritic.  The difference
    is purely in the training regime: DR axes are enabled one at a time in
    the same order as CRA, but the *entire* network continues training with
    all parameters unfrozen (no cascade, no frozen stages).

    This baseline isolates whether CRA's gains come from:
      (a) the axis-decomposed curriculum schedule, or
      (b) the modular frozen-cascade architecture.
    If CurriculumDR matches CRA → the architecture is not necessary.
    If CurriculumDR << CRA   → the frozen cascade is the key ingredient.
    """
    pass


# ======================================================================
# RMAPolicy — single adaptation module for all axes
# ======================================================================

class RMAPolicy(nn.Module):
    """Rapid Motor Adaptation baseline.

    A single GRU adaptation encoder produces a latent that is used to
    *modulate* a conditioned actor.  This is the standard RMA approach
    where one module must jointly identify all environment parameters.

    Architecture
    ------------
    Encoder : GRU over (obs, action) history  ->  z ∈ R^latent_dim
    Actor   : MLP(obs, z) -> action_mean
    Critic  : MLP(obs, z) -> value
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 32,
        encoder_hidden_dim: int = 256,
        encoder_num_layers: int = 2,
        actor_hidden_dims: list[int] | None = None,
        critic_hidden_dims: list[int] | None = None,
        window_size: int = 50,
        init_noise_std: float = 1.0,
    ) -> None:
        super().__init__()
        if actor_hidden_dims is None:
            actor_hidden_dims = [512, 256, 128]
        if critic_hidden_dims is None:
            critic_hidden_dims = [512, 256, 128]

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.window_size = window_size

        # Single adaptation encoder (must learn ALL axes)
        self.encoder = AdaptationEncoder(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=encoder_hidden_dim,
            latent_dim=latent_dim,
            num_layers=encoder_num_layers,
            window_size=window_size,
        )

        # Conditioned actor
        self.actor = _build_mlp(
            obs_dim + latent_dim, action_dim, actor_hidden_dims, nn.ELU
        )

        # Conditioned critic
        self.critic = _build_mlp(
            obs_dim + latent_dim, 1, critic_hidden_dims, nn.ELU
        )

        self.log_std = nn.Parameter(
            torch.ones(action_dim) * torch.tensor(init_noise_std).log()
        )

        self._init_output_weights()

    def _init_output_weights(self) -> None:
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor[-1].bias)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.zeros_(self.critic[-1].bias)

    def forward(
        self,
        obs: torch.Tensor,
        obs_history: torch.Tensor,
        action_history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(obs_history, action_history)
        inp = torch.cat([obs, z], dim=-1)
        action_mean = self.actor(inp)
        action_std = self.log_std.exp().expand_as(action_mean)
        value = self.critic(inp)
        return action_mean, action_std, value

    def get_action(
        self,
        obs: torch.Tensor,
        obs_history: torch.Tensor,
        action_history: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean, action_std, value = self.forward(obs, obs_history, action_history)
        dist = Normal(action_mean, action_std)
        action = action_mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        obs_history: torch.Tensor,
        action_history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean, action_std, value = self.forward(obs, obs_history, action_history)
        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value
