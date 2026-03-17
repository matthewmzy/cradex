#!/usr/bin/env python3
"""Evaluation script for trained CRA / baseline policies.

Usage:
    python scripts/eval.py --checkpoint outputs/cra_42/checkpoints/stage_4_friction.pt \
                           --method cra --num-episodes 200
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cra.envs.shadow_hand_rotation import ShadowHandRotation, RotationEnvConfig
from cra.models.cra_policy import CRAPolicy, CRAStageConfig
from cra.models.baselines import FullDRPolicy, CurriculumDRPolicy, RMAPolicy
from cra.algo.rollout_buffer import HistoryBuffer
from cra.utils.checkpoint import load_checkpoint


def evaluate_cra(
    checkpoint_path: str,
    env: ShadowHandRotation,
    num_episodes: int = 200,
    device: str = "cuda:0",
    axis_order: list[str] | None = None,
) -> dict[str, float]:
    """Load and evaluate a CRA policy."""
    if axis_order is None:
        axis_order = ["gravity", "object_mass", "friction"]

    dev = torch.device(device)

    # Reconstruct policy architecture
    policy = CRAPolicy(
        obs_dim=env.cfg.obs_dim,
        action_dim=env.cfg.action_dim,
    ).to(dev)
    policy.init_base()

    for name in axis_order:
        policy.add_stage(CRAStageConfig(name=name))

    # Load weights
    meta = load_checkpoint(policy, checkpoint_path, device=dev, strict=False)
    print(f"Loaded checkpoint: stage={meta.get('stage', '?')}, "
          f"iter={meta.get('iteration', '?')}")

    policy.to(dev)
    policy.eval()

    # Enable all DR in eval env
    for name in axis_order:
        env.dr_manager.enable_axis(name)

    # Run evaluation
    hist = HistoryBuffer(
        num_envs=env.num_envs,
        obs_dim=env.cfg.obs_dim,
        action_dim=env.cfg.action_dim,
        window_size=policy.window_size,
        device=dev,
    )

    obs = env.reset()
    hist.reset(torch.arange(env.num_envs, device=dev))

    total_reward = 0.0
    total_success = 0.0
    episodes_done = 0
    ep_rewards = torch.zeros(env.num_envs, device=dev)

    while episodes_done < num_episodes:
        with torch.no_grad():
            oh, ah = hist.get()
            action, _, _ = policy.get_action(obs, oh, ah, deterministic=True)

        obs, reward, done, extras = env.step(action)
        ep_rewards += reward
        hist.push(obs, action)

        done_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(done_ids) > 0:
            for d in done_ids:
                total_reward += ep_rewards[d].item()
                episodes_done += 1
            total_success += extras.get("success_rate", 0.0) * len(done_ids)
            ep_rewards[done_ids] = 0
            hist.reset(done_ids)

    results = {
        "mean_reward": total_reward / max(episodes_done, 1),
        "success_rate": total_success / max(episodes_done, 1),
        "episodes": episodes_done,
    }

    print("\nEvaluation Results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    return results


def evaluate_baseline(
    checkpoint_path: str,
    env: ShadowHandRotation,
    method: str = "full_dr",
    num_episodes: int = 200,
    device: str = "cuda:0",
) -> dict[str, float]:
    """Load and evaluate a baseline policy."""
    dev = torch.device(device)

    if method == "full_dr" or method == "curriculum_dr":
        policy_cls = CurriculumDRPolicy if method == "curriculum_dr" else FullDRPolicy
        policy = policy_cls(
            obs_dim=env.cfg.obs_dim,
            action_dim=env.cfg.action_dim,
        ).to(dev)
    elif method == "rma":
        policy = RMAPolicy(
            obs_dim=env.cfg.obs_dim,
            action_dim=env.cfg.action_dim,
        ).to(dev)
    else:
        raise ValueError(f"Unknown method: {method}")

    load_checkpoint(policy, checkpoint_path, device=dev, strict=False)
    policy.eval()

    # Enable all DR
    for ax in ["gravity", "object_mass", "friction"]:
        env.dr_manager.enable_axis(ax)

    use_hist = method == "rma"
    hist = None
    if use_hist:
        hist = HistoryBuffer(
            num_envs=env.num_envs,
            obs_dim=env.cfg.obs_dim,
            action_dim=env.cfg.action_dim,
            window_size=policy.window_size,
            device=dev,
        )

    obs = env.reset()
    if hist:
        hist.reset(torch.arange(env.num_envs, device=dev))

    total_reward = 0.0
    total_success = 0.0
    episodes_done = 0
    ep_rewards = torch.zeros(env.num_envs, device=dev)

    while episodes_done < num_episodes:
        with torch.no_grad():
            if use_hist and hist:
                oh, ah = hist.get()
                action, _, _ = policy.get_action(obs, oh, ah, deterministic=True)
            else:
                action, _, _ = policy.get_action(obs, deterministic=True)

        obs, reward, done, extras = env.step(action)
        ep_rewards += reward
        if hist:
            hist.push(obs, action)

        done_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(done_ids) > 0:
            for d in done_ids:
                total_reward += ep_rewards[d].item()
                episodes_done += 1
            total_success += extras.get("success_rate", 0.0) * len(done_ids)
            ep_rewards[done_ids] = 0
            if hist:
                hist.reset(done_ids)

    results = {
        "mean_reward": total_reward / max(episodes_done, 1),
        "success_rate": total_success / max(episodes_done, 1),
        "episodes": episodes_done,
    }
    print("\nEvaluation Results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained policy")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--method", type=str, default="cra",
                        choices=["cra", "full_dr", "rma", "curriculum_dr"])
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--object-type", type=str, default="cube")
    args = parser.parse_args()

    env_cfg = RotationEnvConfig(
        num_envs=args.num_envs,
        headless=args.headless,
        device=args.device,
        object_type=args.object_type,
    )
    env = ShadowHandRotation(env_cfg)

    if args.method == "cra":
        evaluate_cra(args.checkpoint, env, args.num_episodes, args.device)
    else:
        evaluate_baseline(
            args.checkpoint, env, args.method, args.num_episodes, args.device,
        )

    env.close()


if __name__ == "__main__":
    main()
