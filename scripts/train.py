#!/usr/bin/env python3
"""Main training entry point for CRA and baselines.

Usage:
    # CRA training (multi-stage)
    python scripts/train.py --method cra --config configs/method/cra.yaml

    # Full DR baseline
    python scripts/train.py --method full_dr --config configs/method/full_dr.yaml

    # RMA baseline
    python scripts/train.py --method rma --config configs/method/rma.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cra.envs.shadow_hand_rotation import ShadowHandRotation, RotationEnvConfig
from cra.trainer.cra_trainer import CRATrainer, CRATrainerConfig
from cra.trainer.baseline_trainer import BaselineTrainer, BaselineTrainerConfig
from cra.algo.ppo import PPOConfig


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def merge_config(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = merge_config(result[k], v)
        else:
            result[k] = v
    return result


def build_env(cfg: dict) -> ShadowHandRotation:
    """Build the training environment from config."""
    env_cfg = RotationEnvConfig(
        num_envs=cfg.get("num_envs", 4096),
        episode_length=cfg.get("episode_length", 200),
        headless=cfg.get("headless", True),
        device=cfg.get("device", "cuda:0"),
        object_type=cfg.get("object_type", "cube"),
        object_asset_file=cfg.get("object_asset_file", ""),
        success_tolerance=cfg.get("success_tolerance", 0.1),
        rotation_reward_scale=cfg.get("rotation_reward_scale", 1.0),
        fall_dist=cfg.get("fall_dist", 0.3),
        action_penalty_scale=cfg.get("action_penalty_scale", 0.02),
    )
    return ShadowHandRotation(env_cfg)


def build_ppo_config(cfg: dict) -> PPOConfig:
    """Build PPO config from dict."""
    return PPOConfig(
        lr=cfg.get("lr", 3e-4),
        clip_ratio=cfg.get("clip_ratio", 0.2),
        value_loss_coef=cfg.get("value_loss_coef", 0.5),
        entropy_coef=cfg.get("entropy_coef", 0.0),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        num_epochs=cfg.get("num_epochs", 5),
        mini_batch_size=cfg.get("mini_batch_size", 4096),
        gamma=cfg.get("gamma", 0.99),
        gae_lambda=cfg.get("gae_lambda", 0.95),
        lr_schedule=cfg.get("lr_schedule", "fixed"),
    )


def main():
    parser = argparse.ArgumentParser(description="CRA Dexterous Manipulation Training")
    parser.add_argument("--method", type=str, default="cra",
                        choices=["cra", "full_dr", "rma"],
                        help="Training method")
    parser.add_argument("--config", type=str, default="",
                        help="Path to YAML config file")
    parser.add_argument("--task-config", type=str, default="",
                        help="Path to task-specific config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--experiment-name", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--resume", type=str, default="",
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Load configs
    method_cfg = load_config(args.config)
    task_cfg = load_config(args.task_config)
    cfg = merge_config(task_cfg, method_cfg)

    # CLI overrides
    cfg["seed"] = args.seed
    cfg["device"] = args.device
    cfg["num_envs"] = args.num_envs
    cfg["headless"] = args.headless

    exp_name = args.experiment_name or f"{args.method}_{args.seed}"

    # Build environment
    print(f"Building environment with {args.num_envs} envs on {args.device}...")
    env = build_env(cfg)

    # Build eval env (smaller)
    eval_cfg = cfg.copy()
    eval_cfg["num_envs"] = min(256, args.num_envs)
    eval_env = build_env(eval_cfg)

    # Build trainer
    ppo_cfg = build_ppo_config(cfg.get("ppo", {}))

    if args.method == "cra":
        cra_cfg = cfg.get("cra", {})
        trainer_cfg = CRATrainerConfig(
            experiment_name=exp_name,
            output_dir=args.output_dir,
            seed=args.seed,
            base_num_iterations=cra_cfg.get("base_num_iterations", 5000),
            base_num_steps=cra_cfg.get("base_num_steps", 16),
            stage_num_iterations=cra_cfg.get("stage_num_iterations", 3000),
            stage_num_steps=cra_cfg.get("stage_num_steps", 16),
            axis_order=cra_cfg.get("axis_order", [
                "gravity_dir", "gravity_mag", "object_mass", "friction"
            ]),
            stage_encoder_type=cra_cfg.get("encoder_type", "gru"),
            stage_encoder_hidden=cra_cfg.get("encoder_hidden", 128),
            stage_encoder_latent=cra_cfg.get("encoder_latent", 16),
            stage_window_size=cra_cfg.get("window_size", 50),
            ppo=ppo_cfg,
            device=args.device,
            resume_path=args.resume,
        )
        trainer = CRATrainer(trainer_cfg, env, eval_env)
    else:
        baseline_cfg = BaselineTrainerConfig(
            experiment_name=exp_name,
            output_dir=args.output_dir,
            seed=args.seed,
            method=args.method,
            num_iterations=cfg.get("num_iterations", 10000),
            num_steps=cfg.get("num_steps", 16),
            ppo=ppo_cfg,
            device=args.device,
            dr_axes=cfg.get("dr_axes", [
                "gravity_dir", "gravity_mag", "object_mass", "friction"
            ]),
        )
        if args.method == "rma":
            baseline_cfg.rma_latent_dim = cfg.get("rma_latent_dim", 32)
            baseline_cfg.rma_encoder_hidden = cfg.get("rma_encoder_hidden", 256)
            baseline_cfg.rma_window_size = cfg.get("rma_window_size", 50)
        trainer = BaselineTrainer(baseline_cfg, env, eval_env)

    # Train
    print(f"Starting {args.method} training: {exp_name}")
    trainer.train()


if __name__ == "__main__":
    main()
