#!/usr/bin/env python3
"""Visualize CRA adaptation module latent representations.

Generates:
  - t-SNE plots of each stage's adaptation latent, colored by the
    corresponding DR parameter value
  - Linear probe accuracy (R^2) for predicting DR params from latents
  - Ablation bar chart showing performance impact of disabling each stage

Usage:
    python scripts/visualize_adaptation.py \
        --checkpoint outputs/cra_42/checkpoints/stage_4_friction.pt \
        --num-steps 500 --output-dir outputs/cra_42/analysis
"""

from __future__ import annotations

import argparse
import os
import sys
import json

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cra.envs.shadow_hand_rotation import ShadowHandRotation, RotationEnvConfig
from cra.models.cra_policy import CRAPolicy, CRAStageConfig
from cra.utils.checkpoint import load_checkpoint
from cra.utils.analysis import (
    collect_latents,
    tsne_latents,
    linear_probe,
    ablation_study,
)


def main():
    parser = argparse.ArgumentParser(description="Analyze CRA adaptation modules")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-dir", type=str, default="outputs/analysis")
    parser.add_argument("--axis-order", nargs="+",
                        default=["gravity_dir", "gravity_mag",
                                 "object_mass", "friction"])
    parser.add_argument("--run-ablation", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dev = torch.device(args.device)

    # Build env with all DR enabled
    env_cfg = RotationEnvConfig(
        num_envs=args.num_envs,
        headless=True,
        device=args.device,
    )
    env = ShadowHandRotation(env_cfg)
    for ax in args.axis_order:
        env.dr_manager.enable_axis(ax)

    # Rebuild and load policy
    policy = CRAPolicy(
        obs_dim=env.cfg.obs_dim,
        action_dim=env.cfg.action_dim,
    ).to(dev)
    policy.init_base()
    for name in args.axis_order:
        policy.add_stage(CRAStageConfig(name=name))
    load_checkpoint(policy, args.checkpoint, device=dev, strict=False)
    policy.to(dev)

    # --- Collect latents ---
    print("Collecting adaptation latents...")
    data = collect_latents(policy, env, num_steps=args.num_steps, device=dev)

    probe_results = {}

    for stage_name, stage_data in data.items():
        print(f"\n--- Stage: {stage_name} ---")
        latents = stage_data["latents"]
        print(f"  Latent tensor shape: {latents.shape}")

        # t-SNE visualization
        labels = None
        if "dr_params" in stage_data:
            dr = stage_data["dr_params"]
            if dr.shape[-1] == 1:
                labels = dr.squeeze(-1)
            else:
                labels = dr.norm(dim=-1)  # use magnitude for multi-dim

        tsne_path = os.path.join(args.output_dir, f"tsne_{stage_name}.png")
        print(f"  Computing t-SNE -> {tsne_path}")
        tsne_latents(latents, labels=labels, save_path=tsne_path)

        # Linear probe
        if "dr_params" in stage_data:
            dr = stage_data["dr_params"]
            print(f"  Running linear probe...")
            result = linear_probe(latents, dr)
            probe_results[stage_name] = result
            print(f"  Train R²={result['train_r2']:.4f}, "
                  f"Test R²={result['test_r2']:.4f}, "
                  f"MSE={result['test_mse']:.6f}")

    # Save probe results
    probe_path = os.path.join(args.output_dir, "linear_probe_results.json")
    with open(probe_path, "w") as f:
        json.dump(probe_results, f, indent=2)
    print(f"\nLinear probe results saved to: {probe_path}")

    # --- Ablation study ---
    if args.run_ablation:
        print("\nRunning ablation study (disabling stages one at a time)...")
        ablation = ablation_study(policy, env, num_episodes=100, device=dev)

        print("\nAblation Results:")
        for label, metrics in ablation.items():
            print(f"  {label:30s} | reward={metrics['reward']:.3f} | "
                  f"success={metrics['success_rate']:.3f}")

        ablation_path = os.path.join(args.output_dir, "ablation_results.json")
        with open(ablation_path, "w") as f:
            json.dump(ablation, f, indent=2)

        # Plot ablation bar chart
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            labels = list(ablation.keys())
            rewards = [ablation[l]["reward"] for l in labels]
            successes = [ablation[l]["success_rate"] for l in labels]

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].barh(labels, rewards)
            axes[0].set_xlabel("Mean Reward")
            axes[0].set_title("Ablation: Reward")
            axes[1].barh(labels, successes)
            axes[1].set_xlabel("Success Rate")
            axes[1].set_title("Ablation: Success Rate")
            fig.tight_layout()
            fig.savefig(
                os.path.join(args.output_dir, "ablation_plot.png"),
                dpi=150, bbox_inches="tight",
            )
            plt.close(fig)
            print(f"Ablation plot saved to {args.output_dir}/ablation_plot.png")
        except ImportError:
            print("matplotlib not available, skipping ablation plot.")

    env.close()
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
