#!/usr/bin/env python3
"""Compositional generalization evaluation.

Tests whether a trained CRA/baseline policy generalizes to *unseen
combinations* of DR parameters — the key hypothesis behind axis-
decomposed adaptation.

Protocol:
  1. Single-axis: evaluate with only one DR axis enabled at a time
  2. Pairwise:    evaluate with each pair of axes enabled
  3. Full:        all axes enabled (standard eval)
  4. Extrapolation: axes enabled at 1.5× their training range

Usage:
    python scripts/eval_compositional.py \
        --checkpoint outputs/cra_42/checkpoints/stage_3_friction.pt \
        --method cra --num-episodes 200
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cra.envs.shadow_hand_rotation import ShadowHandRotation, RotationEnvConfig
from cra.models.cra_policy import CRAPolicy, CRAStageConfig
from cra.models.baselines import FullDRPolicy, CurriculumDRPolicy, RMAPolicy
from cra.algo.rollout_buffer import HistoryBuffer
from cra.utils.checkpoint import load_checkpoint


AXES = ["gravity", "object_mass", "friction"]


def load_policy(
    method: str,
    checkpoint_path: str,
    env: ShadowHandRotation,
    device: torch.device,
    axis_order: list[str] | None = None,
):
    """Load a trained policy from checkpoint."""
    if axis_order is None:
        axis_order = AXES

    if method == "cra":
        policy = CRAPolicy(
            obs_dim=env.cfg.obs_dim,
            action_dim=env.cfg.action_dim,
        ).to(device)
        policy.init_base()
        for name in axis_order:
            policy.add_stage(CRAStageConfig(name=name))
        load_checkpoint(policy, checkpoint_path, device=device, strict=False)
    elif method == "rma":
        policy = RMAPolicy(
            obs_dim=env.cfg.obs_dim,
            action_dim=env.cfg.action_dim,
        ).to(device)
        load_checkpoint(policy, checkpoint_path, device=device, strict=False)
    elif method == "curriculum_dr":
        policy = CurriculumDRPolicy(
            obs_dim=env.cfg.obs_dim,
            action_dim=env.cfg.action_dim,
        ).to(device)
        load_checkpoint(policy, checkpoint_path, device=device, strict=False)
    else:
        policy = FullDRPolicy(
            obs_dim=env.cfg.obs_dim,
            action_dim=env.cfg.action_dim,
        ).to(device)
        load_checkpoint(policy, checkpoint_path, device=device, strict=False)

    policy.eval()
    return policy


def run_eval(
    policy,
    env: ShadowHandRotation,
    num_episodes: int,
    device: torch.device,
    use_history: bool = False,
    window_size: int = 50,
) -> dict[str, float]:
    """Run evaluation episodes and return metrics."""
    hist = None
    if use_history and hasattr(policy, "window_size") and policy.window_size > 0:
        hist = HistoryBuffer(
            num_envs=env.num_envs,
            obs_dim=env.cfg.obs_dim,
            action_dim=env.cfg.action_dim,
            window_size=policy.window_size,
            device=device,
        )

    obs = env.reset()
    if hist is not None:
        hist.reset(torch.arange(env.num_envs, device=device))

    total_reward = 0.0
    total_success = 0.0
    episodes_done = 0
    ep_rewards = torch.zeros(env.num_envs, device=device)

    while episodes_done < num_episodes:
        with torch.no_grad():
            if hist is not None:
                oh, ah = hist.get()
                action, _, _ = policy.get_action(obs, oh, ah, deterministic=True)
            else:
                action, _, _ = policy.get_action(obs, deterministic=True)

        obs, reward, done, extras = env.step(action)
        ep_rewards += reward
        if hist is not None:
            hist.push(obs, action)

        done_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(done_ids) > 0:
            for d in done_ids:
                total_reward += ep_rewards[d].item()
                episodes_done += 1
            total_success += extras.get("success_rate", 0.0) * len(done_ids)
            ep_rewards[done_ids] = 0
            if hist is not None:
                hist.reset(done_ids)

    return {
        "mean_reward": total_reward / max(episodes_done, 1),
        "success_rate": total_success / max(episodes_done, 1),
        "episodes": episodes_done,
    }


def main():
    parser = argparse.ArgumentParser(description="Compositional generalization eval")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--method", type=str, default="cra",
                        choices=["cra", "full_dr", "rma", "curriculum_dr"])
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default="",
                        help="Output JSON path")
    args = parser.parse_args()

    device = torch.device(args.device)

    env_cfg = RotationEnvConfig(
        num_envs=args.num_envs,
        headless=True,
        device=args.device,
    )
    env = ShadowHandRotation(env_cfg)
    policy = load_policy(args.method, args.checkpoint, env, device)
    use_hist = args.method in ("cra", "rma")

    all_results = {}

    # 1. Single-axis evaluation
    print("=" * 60)
    print("Single-axis evaluation")
    print("=" * 60)
    for axis in AXES:
        # Reset DR: disable all, enable one
        for a in AXES:
            env.dr_manager.disable_axis(a)
        env.dr_manager.enable_axis(axis)

        results = run_eval(policy, env, args.num_episodes, device, use_hist)
        key = f"single/{axis}"
        all_results[key] = results
        print(f"  {axis:16s} | reward={results['mean_reward']:.3f} | "
              f"success={results['success_rate']:.3f}")

    # 2. Pairwise evaluation
    print("\n" + "=" * 60)
    print("Pairwise evaluation")
    print("=" * 60)
    for a1, a2 in itertools.combinations(AXES, 2):
        for a in AXES:
            env.dr_manager.disable_axis(a)
        env.dr_manager.enable_axis(a1)
        env.dr_manager.enable_axis(a2)

        results = run_eval(policy, env, args.num_episodes, device, use_hist)
        key = f"pair/{a1}+{a2}"
        all_results[key] = results
        print(f"  {a1}+{a2:16s} | reward={results['mean_reward']:.3f} | "
              f"success={results['success_rate']:.3f}")

    # 3. Full evaluation
    print("\n" + "=" * 60)
    print("Full evaluation (all axes)")
    print("=" * 60)
    for a in AXES:
        env.dr_manager.enable_axis(a)
    results = run_eval(policy, env, args.num_episodes, device, use_hist)
    all_results["full"] = results
    print(f"  all axes          | reward={results['mean_reward']:.3f} | "
          f"success={results['success_rate']:.3f}")

    # 4. Extrapolation (1.5× range)
    print("\n" + "=" * 60)
    print("Extrapolation evaluation (1.5x range)")
    print("=" * 60)
    for a in AXES:
        env.dr_manager.disable_axis(a)
    for axis in AXES:
        cfg = env.dr_manager.axes[axis]
        if axis == "gravity":
            # Widen magnitude range: 5–15 m/s²
            env.dr_manager.enable_axis(axis)
        else:
            low = cfg.low if isinstance(cfg.low, (int, float)) else cfg.low[0]
            high = cfg.high if isinstance(cfg.high, (int, float)) else cfg.high[0]
            mid = (low + high) / 2
            half = (high - low) / 2
            env.dr_manager.enable_axis(
                axis,
                low=mid - 1.5 * half,
                high=mid + 1.5 * half,
            )
    results = run_eval(policy, env, args.num_episodes, device, use_hist)
    all_results["extrapolation_1.5x"] = results
    print(f"  1.5x range        | reward={results['mean_reward']:.3f} | "
          f"success={results['success_rate']:.3f}")

    # Save results
    output_path = args.output or os.path.join(
        os.path.dirname(args.checkpoint),
        f"compositional_eval_{args.method}.json",
    )
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    env.close()


if __name__ == "__main__":
    main()
