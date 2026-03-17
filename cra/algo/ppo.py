"""PPO (Proximal Policy Optimization) with clipped surrogate objective.

This is a clean, self-contained PPO implementation designed for
GPU-parallel dexterous manipulation environments (IsaacGym).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim

from cra.algo.rollout_buffer import RolloutBuffer


@dataclass
class PPOConfig:
    """PPO hyper-parameters."""
    lr: float = 3e-4
    eps: float = 1e-5                  # Adam epsilon
    clip_ratio: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.0
    max_grad_norm: float = 1.0
    num_epochs: int = 5
    mini_batch_size: int = 4096
    target_kl: float | None = None     # early stop if KL exceeds
    clip_value_loss: bool = True
    value_clip_range: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr_schedule: str = "fixed"         # "fixed", "linear", "cosine"
    min_lr: float = 0.0
    orthogonality_coef: float = 0.01   # weight for latent orthogonality loss


class PPO:
    """PPO trainer that works with any policy exposing
    ``evaluate_actions(obs, actions, ...) -> (log_prob, entropy, value)``.
    """

    def __init__(self, cfg: PPOConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.optimizer: optim.Adam | None = None
        self.scheduler: optim.lr_scheduler._LRScheduler | None = None

    def setup_optimizer(
        self,
        parameters: list[nn.Parameter],
        total_steps: int | None = None,
    ) -> None:
        """Create optimizer (and optional LR scheduler)."""
        self.optimizer = optim.Adam(
            parameters, lr=self.cfg.lr, eps=self.cfg.eps
        )
        if self.cfg.lr_schedule == "linear" and total_steps:
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=self.cfg.min_lr / max(self.cfg.lr, 1e-10),
                total_iters=total_steps,
            )
        elif self.cfg.lr_schedule == "cosine" and total_steps:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=self.cfg.min_lr,
            )
        else:
            self.scheduler = None

    def update(
        self,
        policy: nn.Module,
        buffer: RolloutBuffer,
        has_history: bool = False,
    ) -> dict[str, float]:
        """Run PPO update epochs on the collected rollout.

        Parameters
        ----------
        policy      : policy module with ``evaluate_actions`` method.
        buffer      : filled RolloutBuffer.
        has_history : whether to pass obs/action history to the policy.

        Returns
        -------
        Dictionary of training metrics.
        """
        assert self.optimizer is not None, "Call setup_optimizer first"

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_clip_frac = 0.0
        total_orth_loss = 0.0
        num_updates = 0

        for _epoch in range(self.cfg.num_epochs):
            batches = buffer.get_batches(self.cfg.mini_batch_size)

            for batch in batches:
                obs = batch["obs"]
                actions = batch["actions"]
                old_log_probs = batch["old_log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]
                old_values = batch["old_values"]

                # Forward through policy
                eval_kwargs: dict = {"obs": obs, "actions": actions}
                if has_history:
                    eval_kwargs["obs_history"] = batch.get("obs_history")
                    eval_kwargs["action_history"] = batch.get("action_history")
                log_probs, entropy, values = policy.evaluate_actions(**eval_kwargs)
                values = values.squeeze(-1)

                # Policy loss (clipped surrogate)
                ratio = (log_probs - old_log_probs).exp()
                surr1 = ratio * advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio)
                    * advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                if self.cfg.clip_value_loss:
                    value_clipped = old_values + torch.clamp(
                        values - old_values,
                        -self.cfg.value_clip_range,
                        self.cfg.value_clip_range,
                    )
                    vl1 = (values - returns) ** 2
                    vl2 = (value_clipped - returns) ** 2
                    value_loss = 0.5 * torch.max(vl1, vl2).mean()
                else:
                    value_loss = 0.5 * ((values - returns) ** 2).mean()

                # Entropy bonus
                entropy_mean = entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.cfg.value_loss_coef * value_loss
                    - self.cfg.entropy_coef * entropy_mean
                )

                # Orthogonality loss on adaptation latents (CRA only)
                orth_loss_val = 0.0
                if (
                    has_history
                    and self.cfg.orthogonality_coef > 0
                    and hasattr(policy, "_compute_all_latents")
                    and hasattr(policy, "compute_orthogonality_loss")
                    and getattr(policy, "num_stages", 0) >= 2
                ):
                    latents = policy._compute_all_latents(
                        batch.get("obs_history"),
                        batch.get("action_history"),
                    )
                    orth_loss = policy.compute_orthogonality_loss(latents)
                    loss = loss + self.cfg.orthogonality_coef * orth_loss
                    orth_loss_val = orth_loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for group in self.optimizer.param_groups for p in group["params"]],
                    self.cfg.max_grad_norm,
                )
                self.optimizer.step()

                # Metrics
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - ratio.log()).mean()
                    clip_frac = ((ratio - 1.0).abs() > self.cfg.clip_ratio).float().mean()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_mean.item()
                total_approx_kl += approx_kl.item()
                total_clip_frac += clip_frac.item()
                total_orth_loss += orth_loss_val
                num_updates += 1

            # Early stopping on KL (per-epoch average)
            if self.cfg.target_kl is not None:
                epoch_updates = len(batches)
                epoch_kl = sum(
                    ((log_probs - old_log_probs).exp() - 1
                     - (log_probs - old_log_probs)).mean().item()
                    for _ in [None]  # placeholder; use accumulated value
                )
                # Use the most recent epoch's contribution
                if num_updates >= epoch_updates:
                    recent_kl = (total_approx_kl -
                                 (total_approx_kl * (num_updates - epoch_updates) / max(num_updates, 1))
                                 ) / max(epoch_updates, 1)
                else:
                    recent_kl = total_approx_kl / max(num_updates, 1)
                if recent_kl > self.cfg.target_kl:
                    break

        if self.scheduler is not None:
            self.scheduler.step()

        n = max(num_updates, 1)
        current_lr = self.optimizer.param_groups[0]["lr"]
        return {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy": total_entropy / n,
            "approx_kl": total_approx_kl / n,
            "clip_fraction": total_clip_frac / n,
            "orthogonality_loss": total_orth_loss / n,
            "learning_rate": current_lr,
        }
