"""Baseline trainers for comparison with CRA.

- BaselineTrainer: trains FullDR or RMA policies in a single stage.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import torch

from cra.algo.ppo import PPO, PPOConfig
from cra.algo.rollout_buffer import RolloutBuffer, HistoryBuffer
from cra.envs.base_env import DexterousEnvBase
from cra.models.base_policy import ActorCritic
from cra.models.baselines import FullDRPolicy, RMAPolicy
from cra.utils.logger import Logger
from cra.utils.checkpoint import save_checkpoint


@dataclass
class BaselineTrainerConfig:
    """Training configuration for baselines."""
    experiment_name: str = "fulldr_rotation"
    output_dir: str = "outputs"
    seed: int = 42
    method: str = "full_dr"            # "full_dr" or "rma"
    num_iterations: int = 10000
    num_steps: int = 16
    ppo: PPOConfig = field(default_factory=PPOConfig)
    log_interval: int = 10
    save_interval: int = 500
    eval_interval: int = 200
    device: str = "cuda:0"
    # RMA-specific
    rma_latent_dim: int = 32
    rma_encoder_hidden: int = 256
    rma_window_size: int = 50
    # DR axes to enable (all at once for full_dr/rma)
    dr_axes: list[str] = field(default_factory=lambda: [
        "gravity_dir", "gravity_mag", "object_mass", "friction",
    ])


class BaselineTrainer:
    """Single-stage trainer for FullDR / RMA baselines."""

    def __init__(
        self,
        cfg: BaselineTrainerConfig,
        env: DexterousEnvBase,
        eval_env: DexterousEnvBase | None = None,
    ) -> None:
        self.cfg = cfg
        self.env = env
        self.eval_env = eval_env
        self.device = torch.device(cfg.device)

        self.run_dir = os.path.join(cfg.output_dir, cfg.experiment_name)
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "checkpoints"), exist_ok=True)

        self.logger = Logger(self.run_dir, cfg.experiment_name)

        # Enable all DR axes at once
        for axis in cfg.dr_axes:
            env.dr_manager.enable_axis(axis)
        self.logger.log_text(env.dr_manager.summary())

        # Build policy
        self.use_history = False
        if cfg.method == "full_dr":
            self.policy = FullDRPolicy(
                obs_dim=env.cfg.obs_dim,
                action_dim=env.cfg.action_dim,
            ).to(self.device)
        elif cfg.method == "rma":
            self.policy = RMAPolicy(
                obs_dim=env.cfg.obs_dim,
                action_dim=env.cfg.action_dim,
                latent_dim=cfg.rma_latent_dim,
                encoder_hidden_dim=cfg.rma_encoder_hidden,
                window_size=cfg.rma_window_size,
            ).to(self.device)
            self.use_history = True
        else:
            raise ValueError(f"Unknown baseline method: {cfg.method}")

        self.ppo = PPO(cfg.ppo, self.device)
        self.ppo.setup_optimizer(
            list(self.policy.parameters()),
            total_steps=cfg.num_iterations,
        )

        self.history_buf: HistoryBuffer | None = None
        if self.use_history:
            self.history_buf = HistoryBuffer(
                num_envs=env.num_envs,
                obs_dim=env.cfg.obs_dim,
                action_dim=env.cfg.action_dim,
                window_size=cfg.rma_window_size,
                device=self.device,
            )

        torch.manual_seed(cfg.seed)
        self.total_timesteps = 0

    def train(self) -> None:
        """Single-stage training loop."""
        self.logger.log_text(
            f"Baseline training: {self.cfg.method} | "
            f"{self.cfg.num_iterations} iterations"
        )

        buffer = RolloutBuffer(
            num_envs=self.env.num_envs,
            num_steps=self.cfg.num_steps,
            obs_dim=self.env.cfg.obs_dim,
            action_dim=self.env.cfg.action_dim,
            device=self.device,
            gamma=self.cfg.ppo.gamma,
            gae_lambda=self.cfg.ppo.gae_lambda,
            history_obs_dim=self.env.cfg.obs_dim if self.use_history else 0,
            history_action_dim=self.env.cfg.action_dim if self.use_history else 0,
            history_window=self.cfg.rma_window_size if self.use_history else 0,
        )

        obs = self.env.reset()
        if self.history_buf is not None:
            self.history_buf.reset(
                torch.arange(self.env.num_envs, device=self.device)
            )

        for iteration in range(self.cfg.num_iterations):
            iter_start = time.time()
            buffer.reset()
            self.policy.eval()

            for step in range(self.cfg.num_steps):
                with torch.no_grad():
                    if self.use_history and self.history_buf is not None:
                        oh, ah = self.history_buf.get()
                        action, log_prob, value = self.policy.get_action(
                            obs, oh, ah,
                        )
                    else:
                        action, log_prob, value = self.policy.get_action(obs)

                next_obs, reward, done, extras = self.env.step(action)

                if self.use_history and self.history_buf is not None:
                    ohs, ahs = self.history_buf.get()
                    buffer.insert(
                        obs, action, reward, done, log_prob, value, ohs, ahs,
                    )
                    self.history_buf.push(obs, action)
                    reset_ids = done.nonzero(as_tuple=False).squeeze(-1)
                    self.history_buf.reset(reset_ids)
                else:
                    buffer.insert(obs, action, reward, done, log_prob, value)

                obs = next_obs
                self.total_timesteps += self.env.num_envs

            # Bootstrap
            with torch.no_grad():
                if self.use_history and self.history_buf is not None:
                    oh, ah = self.history_buf.get()
                    _, _, next_val = self.policy.get_action(obs, oh, ah)
                else:
                    _, _, next_val = self.policy.get_action(obs)
            buffer.compute_gae(next_val.squeeze(-1))

            self.policy.train()
            metrics = self.ppo.update(
                self.policy, buffer, has_history=self.use_history,
            )

            iter_time = time.time() - iter_start
            if (iteration + 1) % self.cfg.log_interval == 0:
                fps = (self.cfg.num_steps * self.env.num_envs) / iter_time
                metrics.update({
                    "reward_mean": buffer.rewards.mean().item(),
                    "fps": fps,
                    "total_timesteps": self.total_timesteps,
                })
                if extras:
                    metrics.update({
                        f"env/{k}": v for k, v in extras.items()
                        if isinstance(v, (int, float))
                    })
                self.logger.log_metrics(metrics, step=self.total_timesteps)
                self.logger.log_text(
                    f"  iter {iteration + 1}/{self.cfg.num_iterations} | "
                    f"reward={metrics['reward_mean']:.3f} | fps={fps:.0f}"
                )

            if (iteration + 1) % self.cfg.save_interval == 0:
                save_checkpoint(
                    self.policy,
                    os.path.join(
                        self.run_dir, "checkpoints",
                        f"iter_{iteration + 1}.pt",
                    ),
                )

        self.logger.close()
