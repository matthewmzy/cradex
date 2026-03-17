"""CRA composite policy: the full Cascaded Residual Adaptation policy.

This module orchestrates:
  - A frozen base policy (Stage 0)
  - N cascaded (AdaptationEncoder, ResidualHead) pairs
  - A trainable critic for whichever stage is currently being optimized

Training proceeds stage by stage.  At stage i:
  - Stages 0 .. i-1 are frozen.
  - Stage i's encoder + residual head are trained.
  - A fresh critic is trained alongside.

At inference all residuals are computed in parallel (additive structure):
    a = clip( pi_0(o) + sum_i rho_i(o, z_i),  a_lo, a_hi )
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from cra.models.base_policy import ActorCritic, _build_mlp
from cra.models.adaptation_encoder import AdaptationEncoder, Conv1DAdaptationEncoder
from cra.models.residual_head import ResidualHead


@dataclass
class CRAStageConfig:
    """Configuration for one CRA adaptation stage."""
    name: str = ""                     # e.g. "gravity", "friction"
    encoder_type: str = "gru"          # "gru" or "conv1d"
    encoder_hidden_dim: int = 128
    encoder_latent_dim: int = 16
    encoder_num_layers: int = 2
    residual_hidden_dims: list[int] = field(default_factory=lambda: [256, 128])
    residual_init_scale: float = 0.01
    window_size: int = 50


class StageCritic(nn.Module):
    """Standalone critic used during each CRA training stage.

    Inputs the same observation as the policy plus adaptation latents
    from all active stages, so it can condition its value estimate on
    the full information available to the current composite policy.
    """

    def __init__(
        self,
        obs_dim: int,
        total_latent_dim: int,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        self.net = _build_mlp(obs_dim + total_latent_dim, 1, hidden_dims, nn.ELU)

    def forward(self, obs: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """Return value estimate.

        Parameters
        ----------
        obs     : (B, obs_dim)
        latents : (B, total_latent_dim) — concatenation of all z_i
        """
        x = torch.cat([obs, latents], dim=-1)
        return self.net(x)


class CRAPolicy(nn.Module):
    """Full CRA policy with staged residual adaptation.

    Usage
    -----
    >>> policy = CRAPolicy(obs_dim=100, action_dim=20)
    >>> policy.init_base(actor_hidden=[512,256,128])
    >>> # ... train base with PPO ...
    >>> policy.add_stage(CRAStageConfig(name="gravity"))
    >>> policy.prepare_stage(0)   # freeze base, build critic
    >>> # ... train stage 0 with PPO ...
    >>> policy.add_stage(CRAStageConfig(name="friction"))
    >>> policy.prepare_stage(1)   # freeze base+stage0, build critic
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_low: float = -1.0,
        action_high: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high

        self.base_policy: ActorCritic | None = None
        self.stages = nn.ModuleList()          # list of nn.ModuleDict
        self.stage_configs: list[CRAStageConfig] = []
        self.critic: StageCritic | None = None

        # Exploration noise for the current stage
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self._current_stage_idx: int = -1      # -1 = base training

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def init_base(
        self,
        actor_hidden: list[int] | None = None,
        critic_hidden: list[int] | None = None,
        init_noise_std: float = 1.0,
    ) -> None:
        """Create the Stage-0 base actor-critic."""
        self.base_policy = ActorCritic(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            actor_hidden_dims=actor_hidden,
            critic_hidden_dims=critic_hidden,
            init_noise_std=init_noise_std,
        )

    def add_stage(self, cfg: CRAStageConfig) -> int:
        """Add a new adaptation stage and return its index."""
        if cfg.encoder_type == "gru":
            encoder = AdaptationEncoder(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                hidden_dim=cfg.encoder_hidden_dim,
                latent_dim=cfg.encoder_latent_dim,
                num_layers=cfg.encoder_num_layers,
                window_size=cfg.window_size,
            )
        elif cfg.encoder_type == "conv1d":
            encoder = Conv1DAdaptationEncoder(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                latent_dim=cfg.encoder_latent_dim,
                window_size=cfg.window_size,
            )
        else:
            raise ValueError(f"Unknown encoder type: {cfg.encoder_type}")

        residual = ResidualHead(
            obs_dim=self.obs_dim,
            latent_dim=cfg.encoder_latent_dim,
            action_dim=self.action_dim,
            hidden_dims=cfg.residual_hidden_dims,
            init_scale=cfg.residual_init_scale,
        )

        stage = nn.ModuleDict({"encoder": encoder, "residual": residual})
        self.stages.append(stage)
        self.stage_configs.append(cfg)
        return len(self.stages) - 1

    def prepare_stage(self, stage_idx: int) -> None:
        """Freeze everything before *stage_idx* and build a fresh critic."""
        assert self.base_policy is not None, "Call init_base() first"
        assert 0 <= stage_idx < len(self.stages)

        # Freeze base policy entirely
        for p in self.base_policy.parameters():
            p.requires_grad = False

        # Freeze all previous stages
        for i in range(stage_idx):
            for p in self.stages[i].parameters():
                p.requires_grad = False

        # Ensure current stage is trainable
        for p in self.stages[stage_idx].parameters():
            p.requires_grad = True

        # Re-initialize exploration noise
        nn.init.zeros_(self.log_std)
        self.log_std.requires_grad = True

        # Build a fresh critic that conditions on obs + all latents so far
        total_latent = sum(
            self.stage_configs[i].encoder_latent_dim
            for i in range(stage_idx + 1)
        )
        self.critic = StageCritic(self.obs_dim, total_latent)
        self._current_stage_idx = stage_idx

    def prepare_base_training(self) -> None:
        """Set up for Stage-0 (base policy) training."""
        assert self.base_policy is not None
        for p in self.base_policy.parameters():
            p.requires_grad = True
        self._current_stage_idx = -1

    # ------------------------------------------------------------------
    # Forward / action sampling
    # ------------------------------------------------------------------

    def _compute_all_latents(
        self,
        obs_history: torch.Tensor,
        action_history: torch.Tensor,
        up_to_stage: int | None = None,
    ) -> list[torch.Tensor]:
        """Run all active encoders and return their latent vectors."""
        if up_to_stage is None:
            up_to_stage = len(self.stages) - 1
        latents = []
        for i in range(up_to_stage + 1):
            encoder = self.stages[i]["encoder"]
            z = encoder(obs_history, action_history)
            latents.append(z)
        return latents

    def _compute_all_residuals(
        self,
        obs: torch.Tensor,
        latents: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute and sum all residual actions (parallel-friendly)."""
        total_residual = torch.zeros(
            obs.shape[0], self.action_dim, device=obs.device, dtype=obs.dtype
        )
        for i, z in enumerate(latents):
            residual_head = self.stages[i]["residual"]
            total_residual = total_residual + residual_head(obs, z)
        return total_residual

    def forward_with_latents(
        self,
        obs: torch.Tensor,
        obs_history: torch.Tensor | None = None,
        action_history: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """Compute composite action mean and return latents.

        Returns
        -------
        action_mean : (B, action_dim)
        latents     : list of (B, latent_dim_i) or None if no history
        """
        base_action = self.base_policy.get_action_mean(obs)
        if obs_history is None or len(self.stages) == 0:
            return base_action, None

        latents = self._compute_all_latents(obs_history, action_history)
        residual = self._compute_all_residuals(obs, latents)
        action = base_action + residual

        return torch.clamp(action, self.action_low, self.action_high), latents

    def forward_action_mean(
        self,
        obs: torch.Tensor,
        obs_history: torch.Tensor | None = None,
        action_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the deterministic composite action (no noise)."""
        action_mean, _ = self.forward_with_latents(obs, obs_history, action_history)
        return action_mean

    def get_action(
        self,
        obs: torch.Tensor,
        obs_history: torch.Tensor | None = None,
        action_history: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or deterministically compute) an action.

        Returns
        -------
        action   : (B, action_dim)
        log_prob : (B,)
        value    : (B, 1)
        """
        # --- Base training mode ---
        if self._current_stage_idx == -1:
            return self.base_policy.get_action(obs, deterministic=deterministic)

        # --- Stage training mode ---
        # Compute action mean and latents in a single pass (no redundant GRU calls)
        action_mean, latents = self.forward_with_latents(obs, obs_history, action_history)
        std = self.log_std.exp().expand_as(action_mean)

        from torch.distributions import Normal
        dist = Normal(action_mean, std)
        action = action_mean if deterministic else dist.rsample()
        action = torch.clamp(action, self.action_low, self.action_high)
        log_prob = dist.log_prob(action).sum(dim=-1)

        # Value from the stage critic (reusing latents)
        z_cat = torch.cat(latents, dim=-1)
        value = self.critic(obs, z_cat)

        return action, log_prob, value

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        obs_history: torch.Tensor | None = None,
        action_history: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log_prob, entropy, value for given actions (PPO update).

        Returns
        -------
        log_prob : (B,)
        entropy  : (B,)
        value    : (B, 1)
        """
        if self._current_stage_idx == -1:
            return self.base_policy.evaluate_actions(obs, actions)

        # Single forward pass computes both action_mean and latents
        action_mean, latents = self.forward_with_latents(obs, obs_history, action_history)
        std = self.log_std.exp().expand_as(action_mean)

        from torch.distributions import Normal
        dist = Normal(action_mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        z_cat = torch.cat(latents, dim=-1)
        value = self.critic(obs, z_cat)

        return log_prob, entropy, value

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------

    def compute_orthogonality_loss(
        self, latents: list[torch.Tensor]
    ) -> torch.Tensor:
        """Compute cosine-similarity-based orthogonality loss.

        Penalizes alignment between latent vectors from different stages,
        encouraging each adapter to encode distinct information.

        L_orth = sum_{i<j} cos_sim(z_i, z_j)^2

        All stages should use the same latent_dim for correctness.

        Parameters
        ----------
        latents : list of (B, latent_dim) tensors from each stage.

        Returns
        -------
        loss : scalar tensor (0 when perfectly orthogonal).
        """
        if len(latents) < 2:
            return torch.tensor(0.0, device=latents[0].device)

        loss = torch.tensor(0.0, device=latents[0].device)
        n_pairs = 0
        for i in range(len(latents)):
            for j in range(i + 1, len(latents)):
                zi = latents[i]  # (B, d)
                zj = latents[j]  # (B, d)
                # Cosine similarity squared, averaged over batch
                cos_sim = torch.nn.functional.cosine_similarity(zi, zj, dim=-1)
                loss = loss + (cos_sim ** 2).mean()
                n_pairs += 1

        return loss / max(n_pairs, 1)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Return only the parameters that should be optimized."""
        params = [self.log_std]
        if self._current_stage_idx == -1:
            params.extend(self.base_policy.parameters())
        else:
            idx = self._current_stage_idx
            params.extend(self.stages[idx].parameters())
            if self.critic is not None:
                params.extend(self.critic.parameters())
        return params

    @property
    def num_stages(self) -> int:
        return len(self.stages)

    @property
    def current_stage(self) -> int:
        return self._current_stage_idx

    @property
    def window_size(self) -> int:
        """Max window size across all active stages."""
        if not self.stage_configs:
            return 0
        return max(c.window_size for c in self.stage_configs)

    def state_info(self) -> dict:
        """Return a summary dict for logging."""
        trainable = sum(p.numel() for p in self.get_trainable_parameters())
        total = sum(p.numel() for p in self.parameters())
        return {
            "current_stage": self._current_stage_idx,
            "num_stages": self.num_stages,
            "trainable_params": trainable,
            "total_params": total,
        }
