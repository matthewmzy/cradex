#!/usr/bin/env python3
"""Axis ordering ablation for CRA.

Tests all permutations of the DR axis order to determine whether CRA
performance is sensitive to the order in which axes are introduced.

This is an important ablation for the paper — if order matters strongly,
we need to explain why (e.g., hardest-first is optimal because later
stages can leverage earlier adaptations).

Usage:
    python scripts/ablation_ordering.py --seeds 42 123 456 --device cuda:0
    python scripts/ablation_ordering.py --dry-run  # preview commands
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime


AXES = ["gravity", "object_mass", "friction"]


def main():
    parser = argparse.ArgumentParser(description="CRA axis ordering ablation")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--output-dir", type=str, default="outputs/ablation_ordering")
    parser.add_argument("--base-config", type=str, default="configs/method/cra.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ablation_dir = os.path.join(args.output_dir, f"ordering_{timestamp}")
    os.makedirs(ablation_dir, exist_ok=True)

    script_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "train.py"
    )

    # Generate all permutations of axis order
    permutations = list(itertools.permutations(AXES))
    print(f"Testing {len(permutations)} axis orderings × {len(args.seeds)} seeds "
          f"= {len(permutations) * len(args.seeds)} runs")

    commands = []
    for perm in permutations:
        order_str = "_".join(a[:3] for a in perm)  # e.g., "gra_obj_fri"
        for seed in args.seeds:
            exp_name = f"cra_order_{order_str}_seed{seed}"

            # Write a temporary config with this axis order
            config_dir = os.path.join(ablation_dir, "configs")
            os.makedirs(config_dir, exist_ok=True)
            config_path = os.path.join(config_dir, f"{order_str}.yaml")

            if not os.path.exists(config_path):
                import yaml
                with open(args.base_config) as f:
                    base_cfg = yaml.safe_load(f)
                base_cfg["cra"]["axis_order"] = list(perm)
                with open(config_path, "w") as f:
                    yaml.dump(base_cfg, f, default_flow_style=False)

            cmd = [
                sys.executable, script_path,
                "--method", "cra",
                "--config", config_path,
                "--seed", str(seed),
                "--device", args.device,
                "--num-envs", str(args.num_envs),
                "--experiment-name", exp_name,
                "--output-dir", ablation_dir,
            ]
            commands.append((exp_name, list(perm), cmd))

    # Save manifest
    manifest = {
        "timestamp": timestamp,
        "axes": AXES,
        "permutations": [list(p) for p in permutations],
        "seeds": args.seeds,
        "runs": {name: {"order": order, "cmd": " ".join(cmd)}
                 for name, order, cmd in commands},
    }
    with open(os.path.join(ablation_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    if args.dry_run:
        print("\nDry run — orderings to test:")
        for perm in permutations:
            print(f"  {' → '.join(perm)}")
        print(f"\n{len(commands)} total commands. Use without --dry-run to execute.")
        return

    # Execute
    results = {}
    for i, (name, order, cmd) in enumerate(commands):
        print(f"\n{'=' * 60}")
        print(f"[{i + 1}/{len(commands)}] {name}  ({' → '.join(order)})")
        print(f"{'=' * 60}")
        try:
            subprocess.run(cmd, check=True)
            results[name] = "success"
        except subprocess.CalledProcessError as e:
            results[name] = f"failed (exit {e.returncode})"
        except KeyboardInterrupt:
            results[name] = "interrupted"
            break

    with open(os.path.join(ablation_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAblation complete. Results in {ablation_dir}")


if __name__ == "__main__":
    main()
