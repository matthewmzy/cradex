"""Axis-decomposed Domain Randomization manager.

This module implements the core CRA insight: DR parameters are
decomposed along independent *axes*, and each axis can be enabled
or disabled independently.  During CRA training, axes are enabled
one at a time as new adaptation stages are added.

Supported axes
--------------
- gravity       : unified 3D gravity vector (recommended)
- gravity_dir   : (legacy) direction of gravity (sampled on unit sphere)
- gravity_mag   : (legacy) magnitude of gravity (default 9.81 m/s^2)
- object_mass   : mass of the manipulated object
- friction       : contact friction coefficient
- object_scale  : uniform scaling of object geometry
- stiffness      : PD controller stiffness (kp)
- damping        : PD controller damping (kd)
- object_com     : center-of-mass offset of the object
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class DRAxisConfig:
    """Configuration for a single DR axis."""
    name: str
    enabled: bool = False
    # Range for uniform sampling [low, high].  For multi-dim axes
    # (gravity_dir), these are interpreted per-component.
    low: float | list[float] = 0.0
    high: float | list[float] = 1.0
    # Nominal value used when the axis is disabled.
    nominal: float | list[float] = 0.0
    # Dimensionality (1 for scalar axes, 3 for gravity direction, etc.)
    dim: int = 1
    # Log-uniform sampling (useful for mass, friction)
    log_uniform: bool = False


# ======================================================================
# Pre-defined axis templates
# ======================================================================

def default_axes() -> dict[str, DRAxisConfig]:
    """Return a dictionary of standard DR axes with sensible defaults.

    The unified ``gravity`` axis is recommended over the legacy
    ``gravity_dir`` / ``gravity_mag`` split.  It samples a 3-D gravity
    vector whose magnitude ranges between 7 and 12  m/s² and whose
    direction covers the full sphere.
    """
    return {
        # --- Unified gravity axis (recommended) ---
        "gravity": DRAxisConfig(
            name="gravity",
            enabled=False,
            low=[-1.0, -1.0, -1.0],
            high=[1.0, 1.0, 1.0],
            nominal=[0.0, 0.0, -1.0],
            dim=3,
            # Sampling handled specially: direction on sphere × magnitude
        ),
        # --- Legacy split axes (backward compatibility) ---
        "gravity_dir": DRAxisConfig(
            name="gravity_dir",
            enabled=False,
            low=[-1.0, -1.0, -1.0],
            high=[1.0, 1.0, 1.0],
            nominal=[0.0, 0.0, -1.0],
            dim=3,
        ),
        "gravity_mag": DRAxisConfig(
            name="gravity_mag",
            enabled=False,
            low=7.0,
            high=12.0,
            nominal=9.81,
            dim=1,
        ),
        "object_mass": DRAxisConfig(
            name="object_mass",
            enabled=False,
            low=0.02,
            high=0.5,
            nominal=0.1,
            dim=1,
            log_uniform=True,
        ),
        "friction": DRAxisConfig(
            name="friction",
            enabled=False,
            low=0.3,
            high=2.0,
            nominal=1.0,
            dim=1,
        ),
        "object_scale": DRAxisConfig(
            name="object_scale",
            enabled=False,
            low=0.7,
            high=1.3,
            nominal=1.0,
            dim=1,
        ),
        "kp": DRAxisConfig(
            name="kp",
            enabled=False,
            low=0.5,
            high=2.0,
            nominal=1.0,
            dim=1,
        ),
        "kd": DRAxisConfig(
            name="kd",
            enabled=False,
            low=0.5,
            high=2.0,
            nominal=1.0,
            dim=1,
        ),
        "object_com": DRAxisConfig(
            name="object_com",
            enabled=False,
            low=[-0.01, -0.01, -0.01],
            high=[0.01, 0.01, 0.01],
            nominal=[0.0, 0.0, 0.0],
            dim=3,
        ),
    }


class AxisDRManager:
    """Manages axis-decomposed domain randomization.

    The manager holds the current configuration of all DR axes.
    It can progressively enable axes (one per CRA stage) and
    sample parameter vectors for vectorized environments.

    Usage
    -----
    >>> mgr = AxisDRManager(num_envs=4096, device="cuda")
    >>> mgr.enable_axis("gravity_dir")
    >>> params = mgr.sample()  # returns dict of tensors
    >>> mgr.enable_axis("friction")
    >>> params = mgr.sample()  # now both gravity_dir and friction vary
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str = "cuda",
        axes: dict[str, DRAxisConfig] | None = None,
    ) -> None:
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.axes = axes or default_axes()
        self._enabled_order: list[str] = []

    def enable_axis(
        self,
        name: str,
        low: float | list[float] | None = None,
        high: float | list[float] | None = None,
    ) -> None:
        """Enable a DR axis (optionally override its range)."""
        if name not in self.axes:
            raise KeyError(f"Unknown DR axis: {name}")
        cfg = self.axes[name]
        cfg.enabled = True
        if low is not None:
            cfg.low = low
        if high is not None:
            cfg.high = high
        if name not in self._enabled_order:
            self._enabled_order.append(name)

    def disable_axis(self, name: str) -> None:
        if name in self.axes:
            self.axes[name].enabled = False
            if name in self._enabled_order:
                self._enabled_order.remove(name)

    def enable_all(self) -> None:
        for name in self.axes:
            self.enable_axis(name)

    @property
    def enabled_axes(self) -> list[str]:
        return [n for n in self._enabled_order if self.axes[n].enabled]

    def sample(self, env_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Sample DR parameters for all (or specified) environments.

        Returns
        -------
        Dict mapping axis name -> tensor of shape (N, dim) where N is
        either num_envs or len(env_ids).
        """
        n = self.num_envs if env_ids is None else len(env_ids)
        params: dict[str, torch.Tensor] = {}

        for name, cfg in self.axes.items():
            if cfg.enabled:
                params[name] = self._sample_axis(cfg, n)
            else:
                params[name] = self._nominal(cfg, n)

        return params

    def _sample_axis(self, cfg: DRAxisConfig, n: int) -> torch.Tensor:
        """Sample uniformly (or log-uniformly) within the axis range."""
        # Unified gravity axis: sample direction on sphere × magnitude
        if cfg.name == "gravity":
            direction = torch.randn(n, 3, device=self.device)
            direction = torch.nn.functional.normalize(direction, dim=-1)
            mag_low, mag_high = 7.0, 12.0
            magnitude = mag_low + torch.rand(n, 1, device=self.device) * (mag_high - mag_low)
            return direction * magnitude  # (N, 3) gravity vector

        low = torch.tensor(cfg.low, device=self.device, dtype=torch.float32)
        high = torch.tensor(cfg.high, device=self.device, dtype=torch.float32)

        if low.dim() == 0:
            low = low.unsqueeze(0)
            high = high.unsqueeze(0)

        if cfg.log_uniform:
            log_low = low.clamp(min=1e-6).log()
            log_high = high.clamp(min=1e-6).log()
            u = torch.rand(n, cfg.dim, device=self.device)
            val = (log_low + u * (log_high - log_low)).exp()
        else:
            u = torch.rand(n, cfg.dim, device=self.device)
            val = low + u * (high - low)

        # Legacy gravity direction: normalize to unit sphere
        if cfg.name == "gravity_dir":
            val = torch.nn.functional.normalize(val, dim=-1)

        return val

    def _nominal(self, cfg: DRAxisConfig, n: int) -> torch.Tensor:
        """Return the nominal (fixed) value for a disabled axis."""
        nom = torch.tensor(cfg.nominal, device=self.device, dtype=torch.float32)
        if nom.dim() == 0:
            nom = nom.unsqueeze(0)
        return nom.unsqueeze(0).expand(n, -1).clone()

    def get_gravity_vectors(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return 3-D gravity vectors from sampled params.

        Supports both the unified ``gravity`` axis and the legacy
        ``gravity_dir`` + ``gravity_mag`` split.

        Returns
        -------
        gravity : (N, 3)
        """
        if "gravity" in params and self.axes["gravity"].enabled:
            return params["gravity"]
        # Legacy path
        direction = params["gravity_dir"]   # (N, 3), unit vectors
        magnitude = params["gravity_mag"]   # (N, 1)
        return direction * magnitude

    def summary(self) -> str:
        lines = ["Axis-Decomposed DR Configuration:"]
        for name, cfg in self.axes.items():
            status = "ENABLED" if cfg.enabled else "disabled"
            lines.append(f"  {name:16s} [{status}]  range=[{cfg.low}, {cfg.high}]  nom={cfg.nominal}")
        return "\n".join(lines)
