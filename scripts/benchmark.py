#!/usr/bin/env python3
"""Multi-seed benchmark runner.

Launches training for all methods across multiple seeds and collects
results for statistical comparison.

Usage:
    python scripts/benchmark.py --seeds 42 123 456 789 1337 --device cuda:0
    python scripts/benchmark.py --methods cra full_dr --seeds 42 123
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime


METHODS = {
    "cra": "configs/method/cra.yaml",
    "full_dr": "configs/method/full_dr.yaml",
    "rma": "configs/method/rma.yaml",
    "curriculum_dr": "configs/method/curriculum_dr.yaml",
    "full_dr_large": "configs/method/full_dr_large.yaml",
    "rma_large": "configs/method/rma_large.yaml",
}


def main():
    parser = argparse.ArgumentParser(description="Multi-seed benchmark")
    parser.add_argument("--methods", nargs="+", default=["cra", "full_dr", "rma", "curriculum_dr"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--output-dir", type=str, default="outputs/benchmark")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running them")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark_dir = os.path.join(args.output_dir, f"bench_{timestamp}")
    os.makedirs(benchmark_dir, exist_ok=True)

    script_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "train.py"
    )

    commands = []
    for method in args.methods:
        # Map large variants to base method for train.py
        train_method = method.replace("_large", "")
        config_path = METHODS.get(method)
        if config_path is None:
            print(f"Warning: unknown method '{method}', skipping")
            continue

        for seed in args.seeds:
            exp_name = f"{method}_seed{seed}"
            cmd = [
                sys.executable, script_path,
                "--method", train_method,
                "--config", config_path,
                "--seed", str(seed),
                "--device", args.device,
                "--num-envs", str(args.num_envs),
                "--experiment-name", exp_name,
                "--output-dir", benchmark_dir,
            ]
            commands.append((exp_name, cmd))

    # Print summary
    print(f"Benchmark plan: {len(commands)} runs")
    print(f"  Methods: {args.methods}")
    print(f"  Seeds:   {args.seeds}")
    print(f"  Output:  {benchmark_dir}")
    print()

    # Save manifest
    manifest = {
        "timestamp": timestamp,
        "methods": args.methods,
        "seeds": args.seeds,
        "device": args.device,
        "num_envs": args.num_envs,
        "runs": {name: " ".join(cmd) for name, cmd in commands},
    }
    manifest_path = os.path.join(benchmark_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    if args.dry_run:
        print("Dry run — commands that would be executed:")
        for name, cmd in commands:
            print(f"  [{name}] {' '.join(cmd)}")
        return

    # Execute sequentially
    results = {}
    for i, (name, cmd) in enumerate(commands):
        print(f"\n{'=' * 60}")
        print(f"[{i + 1}/{len(commands)}] Running {name}")
        print(f"{'=' * 60}")
        try:
            proc = subprocess.run(cmd, check=True)
            results[name] = "success"
        except subprocess.CalledProcessError as e:
            print(f"FAILED: {name} (exit code {e.returncode})")
            results[name] = f"failed (exit {e.returncode})"
        except KeyboardInterrupt:
            print(f"\nInterrupted at {name}")
            results[name] = "interrupted"
            break

    # Save results summary
    results_path = os.path.join(benchmark_dir, "results_summary.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBenchmark complete. Results in {benchmark_dir}")


if __name__ == "__main__":
    main()
