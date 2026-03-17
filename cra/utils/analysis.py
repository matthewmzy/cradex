"""Analysis utilities for CRA adaptation modules.

Provides:
  - t-SNE visualization of adaptation latents
  - module ablation (disable individual stages)
  - latent probe (linear probe to predict ground-truth DR params)
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import numpy as np

from cra.models.cra_policy import CRAPolicy


def collect_latents(
    policy: CRAPolicy,
    env,
    num_steps: int = 1000,
    device: torch.device | str = "cuda",
) -> dict[str, dict[str, torch.Tensor]]:
    """Collect adaptation latents and corresponding DR parameters.

    Returns
    -------
    Dict mapping stage_name -> {
        "latents": (num_steps * num_envs, latent_dim),
        "dr_params": dict of (num_steps * num_envs, param_dim),
    }
    """
    from cra.algo.rollout_buffer import HistoryBuffer

    device = torch.device(device)
    policy.eval()

    history_buf = HistoryBuffer(
        num_envs=env.num_envs,
        obs_dim=env.cfg.obs_dim,
        action_dim=env.cfg.action_dim,
        window_size=policy.window_size,
        device=device,
    )

    results: dict[str, dict[str, list]] = {}
    for i, cfg in enumerate(policy.stage_configs):
        results[cfg.name] = {"latents": [], "dr_params": []}

    obs = env.reset()
    history_buf.reset(torch.arange(env.num_envs, device=device))

    for step in range(num_steps):
        with torch.no_grad():
            oh, ah = history_buf.get()
            latents = policy._compute_all_latents(oh, ah)
            action, _, _ = policy.get_action(obs, oh, ah, deterministic=True)

        for i, cfg in enumerate(policy.stage_configs):
            results[cfg.name]["latents"].append(latents[i].cpu())
            if cfg.name in env.dr_params:
                results[cfg.name]["dr_params"].append(
                    env.dr_params[cfg.name].cpu()
                )

        obs, _, done, _ = env.step(action)
        history_buf.push(obs, action)
        reset_ids = done.nonzero(as_tuple=False).squeeze(-1)
        history_buf.reset(reset_ids)

    # Concatenate
    output = {}
    for name, data in results.items():
        output[name] = {
            "latents": torch.cat(data["latents"], dim=0),
        }
        if data["dr_params"]:
            output[name]["dr_params"] = torch.cat(data["dr_params"], dim=0)
    return output


def tsne_latents(
    latents: torch.Tensor,
    labels: torch.Tensor | None = None,
    perplexity: float = 30.0,
    save_path: str | None = None,
) -> np.ndarray:
    """Compute t-SNE embedding of adaptation latents.

    Parameters
    ----------
    latents : (N, latent_dim)
    labels  : (N,) optional scalar labels for coloring

    Returns
    -------
    embedding : (N, 2)
    """
    from sklearn.manifold import TSNE

    X = latents.numpy() if isinstance(latents, torch.Tensor) else latents
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embedding = tsne.fit_transform(X)

    if save_path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        if labels is not None:
            y = labels.numpy() if isinstance(labels, torch.Tensor) else labels
            scatter = ax.scatter(
                embedding[:, 0], embedding[:, 1],
                c=y, cmap="viridis", s=1, alpha=0.5,
            )
            plt.colorbar(scatter, ax=ax)
        else:
            ax.scatter(embedding[:, 0], embedding[:, 1], s=1, alpha=0.5)
        ax.set_title("t-SNE of Adaptation Latents")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return embedding


def linear_probe(
    latents: torch.Tensor,
    targets: torch.Tensor,
    test_fraction: float = 0.2,
) -> dict[str, float]:
    """Train a linear probe to predict DR parameters from latents.

    This measures how well each adaptation module has learned to
    identify its corresponding physical parameter.

    Returns
    -------
    Dict with "train_r2", "test_r2", "test_mse".
    """
    N = latents.shape[0]
    split = int(N * (1 - test_fraction))

    # Shuffle
    perm = torch.randperm(N)
    X = latents[perm].float()
    y = targets[perm].float()
    if y.dim() == 1:
        y = y.unsqueeze(-1)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Closed-form linear regression: w = (X^T X)^{-1} X^T y
    XtX = X_train.T @ X_train + 1e-4 * torch.eye(X_train.shape[1])
    Xty = X_train.T @ y_train
    w = torch.linalg.solve(XtX, Xty)

    y_pred_train = X_train @ w
    y_pred_test = X_test @ w

    def r2_score(y_true, y_pred):
        ss_res = ((y_true - y_pred) ** 2).sum()
        ss_tot = ((y_true - y_true.mean(dim=0)) ** 2).sum()
        return (1 - ss_res / (ss_tot + 1e-8)).item()

    return {
        "train_r2": r2_score(y_train, y_pred_train),
        "test_r2": r2_score(y_test, y_pred_test),
        "test_mse": ((y_test - y_pred_test) ** 2).mean().item(),
    }


def ablation_study(
    policy: CRAPolicy,
    env,
    num_episodes: int = 100,
    device: torch.device | str = "cuda",
) -> dict[str, dict[str, float]]:
    """Measure performance impact of disabling each CRA stage.

    Returns
    -------
    Dict mapping "disable_{name}" -> {"reward": ..., "success_rate": ...}
    Also includes "all_active" as the baseline.
    """
    from cra.algo.rollout_buffer import HistoryBuffer

    device = torch.device(device)
    results = {}

    def _eval_policy(label: str, disabled_stages: set[int]) -> dict[str, float]:
        """Run evaluation with specific stages disabled."""
        policy.eval()
        hist = HistoryBuffer(
            num_envs=env.num_envs,
            obs_dim=env.cfg.obs_dim,
            action_dim=env.cfg.action_dim,
            window_size=max(policy.window_size, 1),
            device=device,
        )

        obs = env.reset()
        hist.reset(torch.arange(env.num_envs, device=device))

        total_reward = 0.0
        total_success = 0.0
        episodes = 0

        while episodes < num_episodes:
            with torch.no_grad():
                oh, ah = hist.get()
                # Compute base action
                base_act = policy.base_policy.get_action_mean(obs)
                # Add only non-disabled residuals
                latents = policy._compute_all_latents(oh, ah)
                residual = torch.zeros_like(base_act)
                for i, z in enumerate(latents):
                    if i not in disabled_stages:
                        residual = residual + policy.stages[i]["residual"](obs, z)
                action = torch.clamp(
                    base_act + residual,
                    policy.action_low,
                    policy.action_high,
                )

            obs, reward, done, extras = env.step(action)
            hist.push(obs, action)

            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            if len(done_ids) > 0:
                total_reward += reward.sum().item()
                total_success += extras.get("success_rate", 0.0) * len(done_ids)
                episodes += len(done_ids)
                hist.reset(done_ids)

        return {
            "reward": total_reward / max(episodes, 1),
            "success_rate": total_success / max(episodes, 1),
        }

    # All stages active
    results["all_active"] = _eval_policy("all_active", disabled_stages=set())

    # Disable each stage individually
    for i, cfg in enumerate(policy.stage_configs):
        results[f"disable_{cfg.name}"] = _eval_policy(
            f"disable_{cfg.name}", disabled_stages={i}
        )

    return results
