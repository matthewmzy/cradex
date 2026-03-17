"""Abstract base class for GPU-vectorized dexterous manipulation environments.

All CRA environments inherit from this class, which provides:
  - IsaacGym simulation lifecycle management
  - Standardized observation / action / reward interface
  - Integration point for AxisDRManager
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import torch

try:
    from isaacgym import gymapi, gymtorch, gymutil
    HAS_ISAACGYM = True
except ImportError:
    HAS_ISAACGYM = False

from cra.envs.axis_dr import AxisDRManager


@dataclass
class EnvConfig:
    """Common environment configuration."""
    num_envs: int = 4096
    env_spacing: float = 0.75
    episode_length: int = 200          # max steps per episode
    control_freq_inv: int = 2          # env steps per sim step ratio
    dt: float = 1.0 / 60.0            # sim timestep
    substeps: int = 2
    device: str = "cuda:0"
    graphics_device_id: int = 0
    headless: bool = True
    # Observation dimensions (filled by subclass)
    obs_dim: int = 0
    action_dim: int = 0
    # Reward
    reward_scale: float = 1.0
    # Object asset
    object_type: str = "cube"          # "cube", "sphere", "cylinder", "ycb"
    object_asset_file: str = ""        # path for custom mesh


class DexterousEnvBase(abc.ABC):
    """Base class for IsaacGym-based dexterous manipulation environments.

    Subclasses must implement:
      - _create_envs()          : load assets and create environments
      - _compute_observations() : fill self.obs_buf
      - _compute_rewards()      : fill self.rew_buf, self.reset_buf
      - _apply_actions()        : apply policy actions to actuators
      - _reset_envs()           : reset specific environment instances
    """

    def __init__(self, cfg: EnvConfig) -> None:
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.device = torch.device(cfg.device)
        self.max_episode_length = cfg.episode_length

        # Will be filled during init
        self.obs_buf: torch.Tensor = None     # (num_envs, obs_dim)
        self.rew_buf: torch.Tensor = None     # (num_envs,)
        self.reset_buf: torch.Tensor = None   # (num_envs,) bool/int
        self.progress_buf: torch.Tensor = None  # (num_envs,) step count
        self.extras: dict = {}

        # Domain randomization manager
        self.dr_manager = AxisDRManager(num_envs=cfg.num_envs, device=cfg.device)
        self.dr_params: dict[str, torch.Tensor] = {}

        if not HAS_ISAACGYM:
            raise ImportError(
                "IsaacGym is required. Install from "
                "https://developer.nvidia.com/isaac-gym"
            )

        self._setup_sim()
        self._create_envs()
        self._prepare_sim()
        self._allocate_buffers()

        # Create viewer for non-headless mode
        self.viewer = None
        if not cfg.headless:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer is None:
                print("Warning: failed to create viewer, running headless")
            else:
                cam_pos = gymapi.Vec3(1.0, 1.0, 1.5)
                cam_target = gymapi.Vec3(0.0, 0.0, 0.6)
                self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    # ------------------------------------------------------------------
    # Simulation setup
    # ------------------------------------------------------------------

    def _setup_sim(self) -> None:
        """Create the IsaacGym simulator instance."""
        self.gym = gymapi.acquire_gym()

        sim_params = gymapi.SimParams()
        sim_params.dt = self.cfg.dt
        sim_params.substeps = self.cfg.substeps
        sim_params.up_axis = gymapi.UP_AXIS_Z

        # PhysX settings
        sim_params.physx.num_threads = 4
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.contact_offset = 0.002
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.bounce_threshold_velocity = 0.2
        sim_params.physx.max_depenetration_velocity = 1000.0
        sim_params.physx.default_buffer_size_multiplier = 5.0
        sim_params.physx.max_gpu_contact_pairs = self.num_envs * 2048
        sim_params.physx.num_subscenes = 0
        sim_params.physx.contact_collection = (
            gymapi.ContactCollection.CC_ALL_SUBSTEPS
        )
        sim_params.use_gpu_pipeline = True

        compute_device = int(str(self.device).split(":")[-1]) if ":" in str(self.device) else 0
        self.sim = self.gym.create_sim(
            compute_device,
            self.cfg.graphics_device_id,
            gymapi.SIM_PHYSX,
            sim_params,
        )
        if self.sim is None:
            raise RuntimeError("Failed to create IsaacGym sim")

        self._sim_params = sim_params

    def _prepare_sim(self) -> None:
        """Finalize simulation setup after environments are created."""
        self.gym.prepare_sim(self.sim)

    def _allocate_buffers(self) -> None:
        """Allocate observation / reward / reset tensors on GPU."""
        self.obs_buf = torch.zeros(
            self.num_envs, self.cfg.obs_dim, device=self.device
        )
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _create_envs(self) -> None:
        """Load assets, create actor handles, finalize."""
        ...

    @abc.abstractmethod
    def _compute_observations(self) -> None:
        """Fill self.obs_buf from simulation state."""
        ...

    @abc.abstractmethod
    def _compute_rewards(self) -> None:
        """Fill self.rew_buf and self.reset_buf."""
        ...

    @abc.abstractmethod
    def _apply_actions(self, actions: torch.Tensor) -> None:
        """Write actions to simulation actuators."""
        ...

    @abc.abstractmethod
    def _reset_envs(self, env_ids: torch.Tensor) -> None:
        """Reset specific environment instances."""
        ...

    def _apply_external_forces(self) -> None:
        """Apply per-env external forces (default: no-op).

        Subclasses that need per-env gravity variation override this.
        """

    # ------------------------------------------------------------------
    # Standard gym-like interface
    # ------------------------------------------------------------------

    def reset(self) -> torch.Tensor:
        """Reset all environments and return initial observations."""
        all_ids = torch.arange(self.num_envs, device=self.device)
        self.dr_params = self.dr_manager.sample()
        self._reset_envs(all_ids)
        self._compute_observations()
        return self.obs_buf

    def step(self, actions: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, dict
    ]:
        """Execute one environment step.

        Returns
        -------
        obs     : (num_envs, obs_dim)
        reward  : (num_envs,)
        done    : (num_envs,)  (1 = episode ended)
        extras  : dict with additional info
        """
        # Apply actions to actuators
        self._apply_actions(actions)

        # Apply per-env external forces (e.g. pseudo-gravity workaround)
        self._apply_external_forces()

        # Step simulation
        for _ in range(self.cfg.control_freq_inv):
            self.gym.simulate(self.sim)

        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # Render if viewer exists
        if self.viewer is not None:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)

        self.progress_buf += 1

        # Compute reward & reset flags
        self._compute_rewards()

        # Check which envs need reset
        reset_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0:
            # Re-sample DR params for reset envs
            new_params = self.dr_manager.sample(reset_ids)
            for k, v in new_params.items():
                self.dr_params[k][reset_ids] = v
            self._reset_envs(reset_ids)

        # Compute new observations
        self._compute_observations()

        done = self.reset_buf.clone().float()
        return self.obs_buf, self.rew_buf, done, self.extras

    def close(self) -> None:
        if hasattr(self, "viewer") and self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        if hasattr(self, "gym") and hasattr(self, "sim"):
            self.gym.destroy_sim(self.sim)
