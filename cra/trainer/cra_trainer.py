"""CRA multi-stage trainer.

Orchestrates the full CRA training pipeline:

  Stage 0 — Base policy training
    Train pi_0 with narrow/fixed DR until convergence.

  Stage 1..N — Cascaded residual adaptation
    For each axis in the specified order:
      1. Enable that DR axis in the environment
      2. Add a new (AdaptationEncoder, ResidualHead) pair
      3. Freeze all previous stages
      4. Train the new stage with PPO until convergence
      5. Checkpoint
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import torch

from cra.algo.ppo import PPO, PPOConfig
from cra.algo.rollout_buffer import RolloutBuffer, HistoryBuffer
from cra.envs.base_env import DexterousEnvBase
from cra.models.cra_policy import CRAPolicy, CRAStageConfig
from cra.utils.logger import Logger
from cra.utils.checkpoint import save_checkpoint, load_checkpoint


@dataclass
class CRATrainerConfig:
    """Full training configuration."""
    # Experiment
    experiment_name: str = "cra_rotation"
    output_dir: str = "outputs"
    seed: int = 42

    # Base training (Stage 0)
    base_num_iterations: int = 5000
    base_num_steps: int = 16          # rollout length per iteration

    # Stage training
    stage_num_iterations: int = 3000
    stage_num_steps: int = 16

    # Axes to train (in order)
    axis_order: list[str] = field(default_factory=lambda: [
        "gravity_dir",
        "gravity_mag",
        "object_mass",
        "friction",
    ])

    # Stage architecture
    stage_encoder_type: str = "gru"
    stage_encoder_hidden: int = 128
    stage_encoder_latent: int = 16
    stage_encoder_layers: int = 2
    stage_residual_hidden: list[int] = field(default_factory=lambda: [256, 128])
    stage_window_size: int = 50

    # PPO
    ppo: PPOConfig = field(default_factory=PPOConfig)

    # Logging
    log_interval: int = 10
    save_interval: int = 500
    eval_interval: int = 200

    # Device
    device: str = "cuda:0"

    # Resume
    resume_path: str = ""


class CRATrainer:
    """Multi-stage CRA training loop."""

    def __init__(
        self,
        cfg: CRATrainerConfig,
        env: DexterousEnvBase,
        eval_env: DexterousEnvBase | None = None,
    ) -> None:
        self.cfg = cfg
        self.env = env
        self.eval_env = eval_env
        self.device = torch.device(cfg.device)

        # Create output dirs
        self.run_dir = os.path.join(cfg.output_dir, cfg.experiment_name)
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "checkpoints"), exist_ok=True)

        # Logger
        self.logger = Logger(
            log_dir=self.run_dir,
            experiment_name=cfg.experiment_name,
        )

        # Build CRA policy
        self.policy = CRAPolicy(
            obs_dim=env.cfg.obs_dim,
            action_dim=env.cfg.action_dim,
            action_low=-1.0,
            action_high=1.0,
        ).to(self.device)

        # PPO
        self.ppo = PPO(cfg.ppo, self.device)

        # History buffer for adaptation encoders
        self.history_buf: HistoryBuffer | None = None

        # Seed
        torch.manual_seed(cfg.seed)

        self.total_timesteps = 0

    def train(self) -> None:
        """Full CRA training pipeline."""
        self.logger.log_text(f"CRA Training — {len(self.cfg.axis_order)} stages")
        self.logger.log_text(f"Axis order: {self.cfg.axis_order}")

        # ============================================================
        # Stage 0: Base policy
        # ============================================================
        self.logger.log_text("=" * 60)
        self.logger.log_text("STAGE 0: Base Policy Training")
        self.logger.log_text("=" * 60)

        self.policy.init_base()
        self.policy.prepare_base_training()
        self.ppo.setup_optimizer(
            self.policy.get_trainable_parameters(),
            total_steps=self.cfg.base_num_iterations,
        )

        self._train_loop(
            num_iterations=self.cfg.base_num_iterations,
            num_steps=self.cfg.base_num_steps,
            stage_name="base",
            use_history=False,
        )

        save_checkpoint(
            self.policy,
            os.path.join(self.run_dir, "checkpoints", "stage_0_base.pt"),
            stage=0,
        )

        # ============================================================
        # Stages 1..N: Cascaded Residual Adaptation
        # ============================================================
        for stage_idx, axis_name in enumerate(self.cfg.axis_order):
            self.logger.log_text("=" * 60)
            self.logger.log_text(
                f"STAGE {stage_idx + 1}: Adaptation for axis '{axis_name}'"
            )
            self.logger.log_text("=" * 60)

            # Enable DR for this axis
            self.env.dr_manager.enable_axis(axis_name)
            self.logger.log_text(self.env.dr_manager.summary())

            # Add adaptation stage
            stage_cfg = CRAStageConfig(
                name=axis_name,
                encoder_type=self.cfg.stage_encoder_type,
                encoder_hidden_dim=self.cfg.stage_encoder_hidden,
                encoder_latent_dim=self.cfg.stage_encoder_latent,
                encoder_num_layers=self.cfg.stage_encoder_layers,
                residual_hidden_dims=self.cfg.stage_residual_hidden,
                window_size=self.cfg.stage_window_size,
            )
            idx = self.policy.add_stage(stage_cfg)
            self.policy.to(self.device)
            self.policy.prepare_stage(idx)

            # Log parameter counts
            info = self.policy.state_info()
            self.logger.log_text(
                f"  Trainable params: {info['trainable_params']:,} / "
                f"Total: {info['total_params']:,}"
            )

            # Fresh optimizer for this stage
            self.ppo.setup_optimizer(
                self.policy.get_trainable_parameters(),
                total_steps=self.cfg.stage_num_iterations,
            )

            # Initialize history buffer
            self.history_buf = HistoryBuffer(
                num_envs=self.env.num_envs,
                obs_dim=self.env.cfg.obs_dim,
                action_dim=self.env.cfg.action_dim,
                window_size=self.policy.window_size,
                device=self.device,
            )

            # Train this stage
            self._train_loop(
                num_iterations=self.cfg.stage_num_iterations,
                num_steps=self.cfg.stage_num_steps,
                stage_name=f"stage_{stage_idx + 1}_{axis_name}",
                use_history=True,
            )

            save_checkpoint(
                self.policy,
                os.path.join(
                    self.run_dir, "checkpoints",
                    f"stage_{stage_idx + 1}_{axis_name}.pt",
                ),
                stage=stage_idx + 1,
            )

        self.logger.log_text("CRA training complete!")
        self.logger.close()

    def _train_loop(
        self,
        num_iterations: int,
        num_steps: int,
        stage_name: str,
        use_history: bool,
    ) -> None:
        """Inner training loop used by both base and stage training."""
        buffer = RolloutBuffer(
            num_envs=self.env.num_envs,
            num_steps=num_steps,
            obs_dim=self.env.cfg.obs_dim,
            action_dim=self.env.cfg.action_dim,
            device=self.device,
            gamma=self.cfg.ppo.gamma,
            gae_lambda=self.cfg.ppo.gae_lambda,
            history_obs_dim=self.env.cfg.obs_dim if use_history else 0,
            history_action_dim=self.env.cfg.action_dim if use_history else 0,
            history_window=self.policy.window_size if use_history else 0,
        )

        obs = self.env.reset()
        if use_history and self.history_buf is not None:
            self.history_buf.reset(torch.arange(self.env.num_envs, device=self.device))

        for iteration in range(num_iterations):
            iter_start = time.time()

            # --- Rollout collection ---
            buffer.reset()
            self.policy.eval()

            for step in range(num_steps):
                with torch.no_grad():
                    if use_history and self.history_buf is not None:
                        obs_hist, act_hist = self.history_buf.get()
                        action, log_prob, value = self.policy.get_action(
                            obs, obs_hist, act_hist,
                        )
                    else:
                        action, log_prob, value = self.policy.get_action(obs)

                next_obs, reward, done, extras = self.env.step(action)

                # Store transition
                if use_history and self.history_buf is not None:
                    obs_hist_snap, act_hist_snap = self.history_buf.get()
                    buffer.insert(
                        obs, action, reward, done, log_prob, value,
                        obs_hist_snap, act_hist_snap,
                    )
                    # Update history buffer
                    self.history_buf.push(obs, action)
                    # Reset history for done envs
                    reset_ids = done.nonzero(as_tuple=False).squeeze(-1)
                    self.history_buf.reset(reset_ids)
                else:
                    buffer.insert(obs, action, reward, done, log_prob, value)

                obs = next_obs
                self.total_timesteps += self.env.num_envs

            # --- Bootstrap value ---
            with torch.no_grad():
                if use_history and self.history_buf is not None:
                    obs_hist, act_hist = self.history_buf.get()
                    _, _, next_value = self.policy.get_action(
                        obs, obs_hist, act_hist,
                    )
                else:
                    _, _, next_value = self.policy.get_action(obs)
            buffer.compute_gae(next_value.squeeze(-1))

            # --- PPO update ---
            self.policy.train()
            metrics = self.ppo.update(
                self.policy, buffer, has_history=use_history,
            )

            # --- Logging ---
            iter_time = time.time() - iter_start
            if (iteration + 1) % self.cfg.log_interval == 0:
                fps = (num_steps * self.env.num_envs) / iter_time
                metrics.update({
                    "reward_mean": buffer.rewards.mean().item(),
                    "reward_std": buffer.rewards.std().item(),
                    "fps": fps,
                    "total_timesteps": self.total_timesteps,
                })
                if extras:
                    metrics.update({
                        f"env/{k}": v for k, v in extras.items()
                        if isinstance(v, (int, float))
                    })
                self.logger.log_metrics(
                    metrics,
                    step=self.total_timesteps,
                    prefix=stage_name,
                )
                self.logger.log_text(
                    f"  [{stage_name}] iter {iteration + 1}/{num_iterations} | "
                    f"reward={metrics['reward_mean']:.3f} | "
                    f"fps={fps:.0f} | "
                    f"kl={metrics['approx_kl']:.4f}"
                )

            # --- Periodic checkpoint ---
            if (iteration + 1) % self.cfg.save_interval == 0:
                save_checkpoint(
                    self.policy,
                    os.path.join(
                        self.run_dir, "checkpoints",
                        f"{stage_name}_iter_{iteration + 1}.pt",
                    ),
                    stage=self.policy.current_stage,
                    iteration=iteration + 1,
                )

            # --- Periodic evaluation ---
            if self.eval_env and (iteration + 1) % self.cfg.eval_interval == 0:
                eval_metrics = self._evaluate(num_episodes=50)
                self.logger.log_metrics(
                    eval_metrics,
                    step=self.total_timesteps,
                    prefix=f"{stage_name}/eval",
                )

    def _evaluate(self, num_episodes: int = 50) -> dict[str, float]:
        """Run deterministic evaluation episodes."""
        if self.eval_env is None:
            return {}

        self.policy.eval()
        total_reward = 0.0
        total_success = 0.0
        episodes_done = 0

        obs = self.eval_env.reset()
        eval_hist = None
        if self.policy.window_size > 0:
            eval_hist = HistoryBuffer(
                num_envs=self.eval_env.num_envs,
                obs_dim=self.eval_env.cfg.obs_dim,
                action_dim=self.eval_env.cfg.action_dim,
                window_size=self.policy.window_size,
                device=self.device,
            )

        ep_rewards = torch.zeros(self.eval_env.num_envs, device=self.device)
        ep_successes = torch.zeros(self.eval_env.num_envs, device=self.device)

        while episodes_done < num_episodes:
            with torch.no_grad():
                if eval_hist is not None:
                    oh, ah = eval_hist.get()
                    action, _, _ = self.policy.get_action(
                        obs, oh, ah, deterministic=True,
                    )
                else:
                    action, _, _ = self.policy.get_action(
                        obs, deterministic=True,
                    )

            obs, reward, done, extras = self.eval_env.step(action)
            ep_rewards += reward

            if eval_hist is not None:
                eval_hist.push(obs, action)

            if "success_rate" in extras:
                ep_successes += extras["success_rate"]

            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            if len(done_ids) > 0:
                for d in done_ids:
                    total_reward += ep_rewards[d].item()
                    total_success += (ep_successes[d] > 0).float().item()
                    episodes_done += 1
                ep_rewards[done_ids] = 0
                ep_successes[done_ids] = 0
                if eval_hist is not None:
                    eval_hist.reset(done_ids)

        return {
            "eval_reward": total_reward / max(episodes_done, 1),
            "eval_success_rate": total_success / max(episodes_done, 1),
            "eval_episodes": episodes_done,
        }
