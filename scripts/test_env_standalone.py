#!/usr/bin/env python3
"""Standalone environment sanity check.

Runs the ShadowHandRotation environment headless with random actions
to verify that DR sampling, observation construction, reward computation,
reset logic, and the per-env gravity workaround all work without errors.

Usage:
    python scripts/test_env_standalone.py
    python scripts/test_env_standalone.py --num-envs 64 --steps 500
    python scripts/test_env_standalone.py --render  # requires non-headless GPU
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cra.envs.shadow_hand_rotation import ShadowHandRotation, RotationEnvConfig
from cra.envs.axis_dr import AxisDRManager


def run_test(args: argparse.Namespace) -> None:
    device = args.device
    print(f"[test_env] Creating ShadowHandRotation ({args.num_envs} envs, device={device})")

    env_cfg = RotationEnvConfig(
        num_envs=args.num_envs,
        headless=not args.render,
        device=device,
        object_type=args.object_type,
    )
    env = ShadowHandRotation(env_cfg)

    print(f"[test_env] obs_dim={env.cfg.obs_dim}, action_dim={env.cfg.action_dim}")
    print(f"[test_env] num_hand_dofs={env.num_hand_dofs}, num_object_bodies={env.num_object_bodies}")

    # ---- Test 1: Reset without DR ----
    print("\n== Test 1: Reset & step without DR ==")
    obs = env.reset()
    assert obs.shape == (args.num_envs, env.cfg.obs_dim), f"obs shape: {obs.shape}"
    assert not torch.isnan(obs).any(), "NaN in observations after reset"
    print(f"  obs mean={obs.mean():.4f}, std={obs.std():.4f}")

    for step in range(20):
        action = torch.rand(args.num_envs, env.cfg.action_dim, device=device) * 2 - 1
        obs, rew, done, extras = env.step(action)
        assert not torch.isnan(obs).any(), f"NaN in obs at step {step}"
        assert not torch.isnan(rew).any(), f"NaN in rew at step {step}"
    print("  PASSED: 20 steps without DR")

    # ---- Test 2: Enable DR axes incrementally ----
    axes = ["gravity", "object_mass", "friction"]
    for axis in axes:
        print(f"\n== Test 2.{axes.index(axis)}: Enable DR axis '{axis}' ==")
        env.dr_manager.enable_axis(axis)
        obs = env.reset()
        assert not torch.isnan(obs).any(), f"NaN after enabling {axis}"

        for step in range(50):
            action = torch.rand(args.num_envs, env.cfg.action_dim, device=device) * 2 - 1
            obs, rew, done, extras = env.step(action)
            if torch.isnan(obs).any():
                nan_mask = torch.isnan(obs).any(dim=0)
                nan_dims = nan_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
                raise AssertionError(
                    f"NaN in obs at step {step} after enabling {axis}, "
                    f"dims={nan_dims}"
                )

        print(f"  obs mean={obs.mean():.4f}, std={obs.std():.4f}")
        print(f"  rew mean={rew.mean():.4f}")
        if "success_rate" in extras:
            print(f"  success_rate={extras['success_rate']:.4f}")
        print(f"  PASSED: 50 steps with {axis}")

    # ---- Test 3: Per-env gravity sanity ----
    print("\n== Test 3: Per-env gravity vector diversity ==")
    gravity = env.per_env_gravity
    gravity_norms = gravity.norm(dim=-1)
    print(f"  gravity norm: mean={gravity_norms.mean():.3f}, "
          f"min={gravity_norms.min():.3f}, max={gravity_norms.max():.3f}")
    assert gravity_norms.min() >= 6.5, f"Gravity magnitude too low: {gravity_norms.min():.3f}"
    assert gravity_norms.max() <= 12.5, f"Gravity magnitude too high: {gravity_norms.max():.3f}"
    print("  PASSED: Gravity magnitudes in [7, 12] range")

    # ---- Test 4: Extended run for stability ----
    print(f"\n== Test 4: Extended run ({args.steps} steps) ==")
    obs = env.reset()
    t0 = time.time()
    total_reward = 0.0
    total_resets = 0

    for step in range(args.steps):
        action = torch.rand(args.num_envs, env.cfg.action_dim, device=device) * 2 - 1
        obs, rew, done, extras = env.step(action)
        total_reward += rew.sum().item()
        total_resets += done.sum().item()

        if (step + 1) % 100 == 0:
            elapsed = time.time() - t0
            fps = ((step + 1) * args.num_envs) / elapsed
            print(f"  step {step + 1}/{args.steps} | fps={fps:.0f} | "
                  f"resets={int(total_resets)}")

    elapsed = time.time() - t0
    fps = (args.steps * args.num_envs) / elapsed
    print(f"  Total: {args.steps} steps, {int(total_resets)} resets, "
          f"{fps:.0f} env-steps/s")
    print("  PASSED: No crashes in extended run")

    # ---- Test 5: DR manager summary ----
    print(f"\n== DR Manager State ==")
    print(env.dr_manager.summary())

    env.close()
    print("\n[test_env] All tests passed!")


def main():
    parser = argparse.ArgumentParser(description="Standalone environment test")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--object-type", type=str, default="cube")
    parser.add_argument("--render", action="store_true", default=False,
                        help="Enable rendering (non-headless)")
    args = parser.parse_args()
    run_test(args)


if __name__ == "__main__":
    main()
