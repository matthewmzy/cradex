"""GPU-resident rollout buffer with integrated history tracking.

The buffer stores transitions for PPO updates.  The HistoryBuffer
maintains a sliding window of (observation, action) pairs needed
by the CRA adaptation encoders.
"""

from __future__ import annotations

import torch


class HistoryBuffer:
    """Rolling history buffer for adaptation encoder inputs.

    Maintains a (num_envs, window_size, dim) tensor that shifts by one
    position every step.  On environment resets the corresponding rows
    are zeroed out.

    This buffer lives on GPU alongside the vectorized environments.
    """

    def __init__(
        self,
        num_envs: int,
        obs_dim: int,
        action_dim: int,
        window_size: int,
        device: torch.device,
    ) -> None:
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.window_size = window_size
        self.device = device

        # (num_envs, window_size, obs_dim)
        self.obs_history = torch.zeros(
            num_envs, window_size, obs_dim, device=device
        )
        # (num_envs, window_size, action_dim)
        self.action_history = torch.zeros(
            num_envs, window_size, action_dim, device=device
        )

    def push(self, obs: torch.Tensor, action: torch.Tensor) -> None:
        """Shift window left and insert new obs-action pair at the end.

        Parameters
        ----------
        obs    : (num_envs, obs_dim)
        action : (num_envs, action_dim)
        """
        # Shift left by one position (no clone needed — non-overlapping
        # copy direction is safe in PyTorch)
        self.obs_history[:, :-1] = self.obs_history[:, 1:]
        self.action_history[:, :-1] = self.action_history[:, 1:]
        # Insert newest at the right end
        self.obs_history[:, -1] = obs
        self.action_history[:, -1] = action

    def reset(self, env_ids: torch.Tensor) -> None:
        """Zero out history for environments that have been reset.

        Parameters
        ----------
        env_ids : 1-D tensor of environment indices to reset.
        """
        if len(env_ids) == 0:
            return
        self.obs_history[env_ids] = 0.0
        self.action_history[env_ids] = 0.0

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return current (obs_history, action_history) snapshots.

        Returns references to the internal tensors.  Callers that need
        to store the result across time steps (e.g. RolloutBuffer) must
        clone explicitly.
        """
        return self.obs_history, self.action_history


class RolloutBuffer:
    """Fixed-length rollout buffer for on-policy PPO training.

    Stores *num_steps* transitions from *num_envs* parallel environments,
    then provides mini-batch iteration for the PPO update.

    All tensors reside on *device* (typically GPU).
    """

    def __init__(
        self,
        num_envs: int,
        num_steps: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        # History dimensions for CRA / RMA (set to 0 to disable)
        history_obs_dim: int = 0,
        history_action_dim: int = 0,
        history_window: int = 0,
    ) -> None:
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.has_history = history_window > 0

        # Core transition storage
        self.observations = torch.zeros(num_steps, num_envs, obs_dim, device=device)
        self.actions = torch.zeros(num_steps, num_envs, action_dim, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, device=device)
        self.dones = torch.zeros(num_steps, num_envs, device=device)
        self.log_probs = torch.zeros(num_steps, num_envs, device=device)
        self.values = torch.zeros(num_steps, num_envs, device=device)
        self.advantages = torch.zeros(num_steps, num_envs, device=device)
        self.returns = torch.zeros(num_steps, num_envs, device=device)

        # Optional history storage for adaptation encoders
        if self.has_history:
            self.obs_histories = torch.zeros(
                num_steps, num_envs, history_window, history_obs_dim, device=device
            )
            self.action_histories = torch.zeros(
                num_steps, num_envs, history_window, history_action_dim, device=device
            )

        self.step = 0

    def insert(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        obs_history: torch.Tensor | None = None,
        action_history: torch.Tensor | None = None,
    ) -> None:
        """Store one transition across all environments."""
        self.observations[self.step] = obs
        self.actions[self.step] = action
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.log_probs[self.step] = log_prob
        self.values[self.step] = value.squeeze(-1)
        if self.has_history and obs_history is not None:
            self.obs_histories[self.step] = obs_history
            self.action_histories[self.step] = action_history
        self.step += 1

    def compute_gae(self, next_value: torch.Tensor) -> None:
        """Compute GAE advantages and discounted returns.

        Parameters
        ----------
        next_value : (num_envs,) — V(s_{T+1}) bootstrap value.
        """
        gae = torch.zeros(self.num_envs, device=self.device)
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_val = next_value
            else:
                next_val = self.values[t + 1]
            delta = (
                self.rewards[t]
                + self.gamma * next_val * (1.0 - self.dones[t])
                - self.values[t]
            )
            gae = delta + self.gamma * self.gae_lambda * (1.0 - self.dones[t]) * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values

    def get_batches(
        self, mini_batch_size: int
    ) -> list[dict[str, torch.Tensor]]:
        """Flatten and shuffle into mini-batches for PPO update.

        Returns
        -------
        List of dicts, each containing tensors of shape (mini_batch_size, ...).
        """
        total = self.num_steps * self.num_envs
        indices = torch.randperm(total, device=self.device)

        flat_obs = self.observations.reshape(total, self.obs_dim)
        flat_act = self.actions.reshape(total, self.action_dim)
        flat_lp = self.log_probs.reshape(total)
        flat_adv = self.advantages.reshape(total)
        flat_ret = self.returns.reshape(total)
        flat_val = self.values.reshape(total)

        flat_obs_hist = None
        flat_act_hist = None
        if self.has_history:
            w = self.obs_histories.shape[2]
            flat_obs_hist = self.obs_histories.reshape(total, w, -1)
            flat_act_hist = self.action_histories.reshape(total, w, -1)

        # Normalize advantages
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

        batches = []
        for start in range(0, total, mini_batch_size):
            end = min(start + mini_batch_size, total)
            idx = indices[start:end]
            batch: dict[str, torch.Tensor] = {
                "obs": flat_obs[idx],
                "actions": flat_act[idx],
                "old_log_probs": flat_lp[idx],
                "advantages": flat_adv[idx],
                "returns": flat_ret[idx],
                "old_values": flat_val[idx],
            }
            if flat_obs_hist is not None:
                batch["obs_history"] = flat_obs_hist[idx]
                batch["action_history"] = flat_act_hist[idx]
            batches.append(batch)
        return batches

    def reset(self) -> None:
        self.step = 0
