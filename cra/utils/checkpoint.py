"""Checkpoint save / load utilities for CRA policies."""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    path: str,
    stage: int = -1,
    iteration: int = 0,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save model checkpoint with metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "stage": stage,
        "iteration": iteration,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(
    model: nn.Module,
    path: str,
    device: torch.device | str = "cpu",
    strict: bool = False,
) -> dict[str, Any]:
    """Load model checkpoint, returning metadata.

    Uses ``strict=False`` by default so that partially-built CRA
    policies (fewer stages than the checkpoint) can still load the
    available weights.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(
        checkpoint["model_state_dict"], strict=strict,
    )
    if missing:
        print(f"  Checkpoint: {len(missing)} missing keys (expected for partial load)")
    if unexpected:
        print(f"  Checkpoint: {len(unexpected)} unexpected keys")
    return {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
