"""Reward functions for dexterous manipulation tasks."""

from __future__ import annotations

import torch


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions (w, x, y, z convention)."""
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of a quaternion (w, x, y, z)."""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_diff_rad(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Angular difference in radians between two quaternions.

    Returns
    -------
    angle : (...,) angular distance in [0, pi].
    """
    q_diff = quat_mul(q1, quat_conjugate(q2))
    # Clamp w to [-1, 1] for numerical stability
    w = q_diff[..., 0].clamp(-1.0, 1.0)
    angle = 2.0 * torch.acos(w.abs())
    return angle


def rotation_reward(
    object_quat: torch.Tensor,
    target_quat: torch.Tensor,
    rot_eps: float = 0.1,
    rot_reward_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute reward for in-hand rotation task.

    Parameters
    ----------
    object_quat : (N, 4) current object orientation (w, x, y, z)
    target_quat : (N, 4) target orientation
    rot_eps     : threshold (rad) for success bonus
    rot_reward_scale : scaling factor

    Returns
    -------
    reward  : (N,)
    success : (N,) boolean success flags
    """
    angle_diff = quat_diff_rad(object_quat, target_quat)  # (N,)

    # Continuous reward: negative angular distance
    reward = -angle_diff * rot_reward_scale

    # Success bonus
    success = (angle_diff < rot_eps).float()
    reward = reward + success * 2.0

    return reward, success


def drop_penalty(
    object_pos: torch.Tensor,
    hand_pos: torch.Tensor,
    threshold: float = 0.5,
    penalty: float = -5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute penalty for dropping the object.

    Returns
    -------
    penalty_val : (N,)
    dropped     : (N,) boolean flags
    """
    dist = torch.norm(object_pos - hand_pos, dim=-1)
    dropped = (dist > threshold).float()
    penalty_val = dropped * penalty
    return penalty_val, dropped


def action_penalty(
    actions: torch.Tensor,
    scale: float = 0.01,
) -> torch.Tensor:
    """L2 penalty on action magnitude for smooth control."""
    return -scale * (actions ** 2).sum(dim=-1)


def fingertip_object_distance_reward(
    fingertip_pos: torch.Tensor,
    object_pos: torch.Tensor,
    scale: float = 0.1,
) -> torch.Tensor:
    """Reward for keeping fingertips close to the object.

    Parameters
    ----------
    fingertip_pos : (N, num_fingers, 3)
    object_pos    : (N, 3)
    """
    # (N, num_fingers)
    dists = torch.norm(
        fingertip_pos - object_pos.unsqueeze(1), dim=-1
    )
    mean_dist = dists.mean(dim=-1)  # (N,)
    return -scale * mean_dist
