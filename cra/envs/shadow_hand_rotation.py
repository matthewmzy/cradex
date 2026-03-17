"""Shadow Hand in-hand rotation environment using IsaacGym.

This environment implements the standard in-hand object rotation task:
  - A Shadow Hand holds an object between its fingertips
  - The goal is to rotate the object to match a target orientation
  - The target changes upon success (continuous rotation)

Observation vector (157-D by default):
  - Hand DOF positions           : 24
  - Hand DOF velocities          : 24
  - Object position (rel hand)   : 3
  - Object quaternion             : 4
  - Object linear velocity       : 3
  - Object angular velocity      : 3
  - Target quaternion             : 4
  - Fingertip positions (5×3)    : 15
  - Fingertip contact forces(5×3): 15
  - Previous action              : 20
  - Gravity vector               : 3
  - DR parameter hints (optional): variable
  -----------------------------------------------
  Total (without hints):           118

Action: 20-D target joint positions for PD controller
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch
import numpy as np

try:
    from isaacgym import gymapi, gymtorch, gymutil
    from isaacgym.torch_utils import (
        quat_from_euler_xyz,
        quat_rotate,
        tensor_clamp,
        to_torch,
        torch_rand_float,
    )
    HAS_ISAACGYM = True
except ImportError:
    HAS_ISAACGYM = False

from cra.envs.base_env import DexterousEnvBase, EnvConfig
from cra.envs import rewards as R
from cra.envs.rewards import xyzw_to_wxyz


# ======================================================================
# Constants for Shadow Hand
# ======================================================================

SHADOW_HAND_NUM_DOFS = 24
SHADOW_HAND_NUM_ACTUATED = 20
NUM_FINGERTIPS = 5

# Default fingertip body names in the Shadow Hand MJCF
FINGERTIP_NAMES = [
    "robot0:ffdistal",
    "robot0:mfdistal",
    "robot0:rfdistal",
    "robot0:lfdistal",
    "robot0:thdistal",
]


@dataclass
class RotationEnvConfig(EnvConfig):
    """Configuration specific to in-hand rotation."""
    obs_dim: int = 118
    action_dim: int = SHADOW_HAND_NUM_ACTUATED
    # Task
    success_tolerance: float = 0.1        # rad
    reach_goal_bonus: float = 250.0
    fall_penalty: float = -50.0
    fall_dist: float = 0.3
    rotation_reward_scale: float = 1.0
    action_penalty_scale: float = 0.02
    consecutive_successes_threshold: int = 50
    # Hand
    hand_asset_file: str = "mjcf/open_ai_assets/hand/shadow_hand.xml"
    # PD gains (multiplied by per-joint defaults)
    kp_scale: float = 1.0
    kd_scale: float = 1.0
    # Target rotation
    target_change_on_success: bool = True
    # Object
    object_type: str = "cube"
    object_asset_file: str = ""


class ShadowHandRotation(DexterousEnvBase):
    """In-hand object rotation with Shadow Hand in IsaacGym.

    This is the primary benchmark environment for CRA experiments.
    """

    def __init__(self, cfg: RotationEnvConfig | None = None) -> None:
        if cfg is None:
            cfg = RotationEnvConfig()
        self.task_cfg = cfg
        super().__init__(cfg)

        # Success tracking
        self.successes = torch.zeros(self.num_envs, device=self.device)
        self.consecutive_successes = torch.zeros(self.num_envs, device=self.device)
        self.prev_actions = torch.zeros(
            self.num_envs, cfg.action_dim, device=self.device
        )
        # Target orientation — stored in IsaacGym (x,y,z,w) convention
        self.target_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.target_quat[:, 3] = 1.0  # identity quaternion in (x,y,z,w)

        # Per-env gravity vectors for external-force workaround.
        # Gravity is a global sim param in IsaacGym, so per-env variation
        # is implemented as an external force on the object rigid body:
        #   F_pseudo = (g_desired - g_global) * m_object
        self.per_env_gravity = torch.zeros(self.num_envs, 3, device=self.device)
        self.per_env_gravity[:, 2] = -9.81  # default
        self.per_env_object_mass = torch.full(
            (self.num_envs,), 0.1, device=self.device
        )
        self._global_gravity = torch.tensor(
            [0.0, 0.0, -9.81], device=self.device
        )

        # Force tensor for apply_rigid_body_force_tensors
        # shape: (num_bodies_total, 3) — only object bodies get nonzero
        self._pseudo_gravity_forces: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Environment creation
    # ------------------------------------------------------------------

    def _create_envs(self) -> None:
        """Load Shadow Hand + object assets, create env instances."""
        # Ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

        # --- Load Shadow Hand asset ---
        asset_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "assets",
        )
        # Try IsaacGym's built-in assets first
        isaacgym_asset_root = os.environ.get(
            "ISAACGYM_ASSET_ROOT",
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "..", "assets", "isaacgym"),
        )

        hand_asset_options = gymapi.AssetOptions()
        hand_asset_options.flip_visual_attachments = False
        hand_asset_options.fix_base_link = True
        hand_asset_options.collapse_fixed_joints = True
        hand_asset_options.disable_gravity = True
        hand_asset_options.thickness = 0.001
        hand_asset_options.angular_damping = 100.0
        hand_asset_options.linear_damping = 100.0

        if os.path.exists(os.path.join(isaacgym_asset_root, self.task_cfg.hand_asset_file)):
            hand_asset_root = isaacgym_asset_root
        else:
            hand_asset_root = asset_root

        self.hand_asset = self.gym.load_asset(
            self.sim, hand_asset_root, self.task_cfg.hand_asset_file,
            hand_asset_options,
        )
        self.num_hand_dofs = self.gym.get_asset_dof_count(self.hand_asset)
        self.num_hand_bodies = self.gym.get_asset_rigid_body_count(self.hand_asset)

        # --- Load object asset ---
        self.object_asset = self._load_object_asset()
        self.num_object_bodies = self.gym.get_asset_rigid_body_count(self.object_asset)

        # --- DOF properties ---
        hand_dof_props = self.gym.get_asset_dof_properties(self.hand_asset)
        self.hand_dof_lower = []
        self.hand_dof_upper = []
        self.hand_dof_default = []
        self.hand_kp = []
        self.hand_kd = []

        for i in range(self.num_hand_dofs):
            self.hand_dof_lower.append(float(hand_dof_props["lower"][i]))
            self.hand_dof_upper.append(float(hand_dof_props["upper"][i]))
            self.hand_dof_default.append(0.0)
            # PD gains
            self.hand_kp.append(float(hand_dof_props["stiffness"][i]) * self.task_cfg.kp_scale)
            self.hand_kd.append(float(hand_dof_props["damping"][i]) * self.task_cfg.kd_scale)

            hand_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            hand_dof_props["stiffness"][i] = self.hand_kp[-1]
            hand_dof_props["damping"][i] = self.hand_kd[-1]
            hand_dof_props["effort"][i] = 0.5

        self.hand_dof_lower = to_torch(self.hand_dof_lower, device=self.device)
        self.hand_dof_upper = to_torch(self.hand_dof_upper, device=self.device)
        self.hand_dof_default = to_torch(self.hand_dof_default, device=self.device)

        # --- Create environments ---
        lower = gymapi.Vec3(-self.cfg.env_spacing, -self.cfg.env_spacing, 0.0)
        upper = gymapi.Vec3(self.cfg.env_spacing, self.cfg.env_spacing, self.cfg.env_spacing)

        self.envs = []
        self.hand_handles = []
        self.object_handles = []
        self.hand_indices = []
        self.object_indices = []

        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, lower, upper, int(math.sqrt(self.num_envs)))

            # Shadow Hand (palm facing up)
            hand_start = gymapi.Transform()
            hand_start.p = gymapi.Vec3(0.0, 0.0, 0.6)
            hand_start.r = gymapi.Quat.from_euler_zyx(0.0, math.pi, 0.0)

            hand_handle = self.gym.create_actor(
                env, self.hand_asset, hand_start, "hand", i, -1, 0
            )
            self.gym.set_actor_dof_properties(env, hand_handle, hand_dof_props)
            hand_idx = self.gym.get_actor_index(env, hand_handle, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

            # Object (above the palm)
            object_start = gymapi.Transform()
            object_start.p = gymapi.Vec3(0.0, 0.0, 0.75)
            object_start.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

            obj_handle = self.gym.create_actor(
                env, self.object_asset, object_start, "object", i, 0, 1
            )
            obj_idx = self.gym.get_actor_index(env, obj_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(obj_idx)

            self.envs.append(env)
            self.hand_handles.append(hand_handle)
            self.object_handles.append(obj_handle)

        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)

        # --- Acquire GPU tensors ---
        self._acquire_tensors()

    def _load_object_asset(self) -> "gymapi.Asset":
        """Load object asset based on object_type config."""
        obj_opts = gymapi.AssetOptions()
        obj_opts.density = 500.0
        obj_opts.override_com = True
        obj_opts.override_inertia = True

        obj_type = self.task_cfg.object_type.lower()
        if obj_type == "cube":
            asset = self.gym.create_box(self.sim, 0.05, 0.05, 0.05, obj_opts)
        elif obj_type == "sphere":
            asset = self.gym.create_sphere(self.sim, 0.03, obj_opts)
        elif obj_type == "cylinder":
            asset = self.gym.create_capsule(self.sim, 0.02, 0.06, obj_opts)
        elif obj_type in ("ycb", "mesh", "custom"):
            asset_root = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "..", "assets", "objects",
            )
            asset_file = self.task_cfg.object_asset_file
            if not asset_file:
                raise ValueError("object_asset_file required for custom objects")
            asset = self.gym.load_asset(self.sim, asset_root, asset_file, obj_opts)
        else:
            raise ValueError(f"Unknown object_type: {obj_type}")

        return asset

    def _acquire_tensors(self) -> None:
        """Wrap IsaacGym GPU state tensors as PyTorch tensors."""
        # DOF state: (num_dofs_total, 2)  [position, velocity]
        dof_state = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state)

        # Root state: (num_actors, 13)  [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_state = gymtorch.wrap_tensor(root_state)

        # Rigid body state: (num_bodies_total, 13)
        rb_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_state = gymtorch.wrap_tensor(rb_state)

        # Net contact forces: (num_bodies_total, 3)
        contact_force = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.contact_force = gymtorch.wrap_tensor(contact_force)

        # Per-hand DOF positions and velocities
        self.hand_dof_pos = self.dof_state[:, 0].view(self.num_envs, -1)[
            :, :self.num_hand_dofs
        ]
        self.hand_dof_vel = self.dof_state[:, 1].view(self.num_envs, -1)[
            :, :self.num_hand_dofs
        ]

        # Root state slices
        self.object_pos = self.root_state[self.object_indices, 0:3]
        self.object_quat = self.root_state[self.object_indices, 3:7]
        self.object_linvel = self.root_state[self.object_indices, 7:10]
        self.object_angvel = self.root_state[self.object_indices, 10:13]
        self.hand_pos = self.root_state[self.hand_indices, 0:3]

        # Fingertip body indices (found by name search)
        self.fingertip_indices = self._find_fingertip_indices()

        # DOF indices for set_dof_state_tensor_indexed
        # Each env has hand DOFs + object DOFs (free body = 6 DOF for root joint).
        # The DOF layout in the global tensor is:
        #   env0: [hand_dofs(24), object_dofs(6)], env1: [...], ...
        dofs_per_env = self.num_hand_dofs + 6  # object has 6 DOFs (free body)
        self.hand_dof_indices = (
            torch.arange(self.num_envs, device=self.device) * dofs_per_env
        ).to(torch.long)

        # Object body indices (sim-global) for external force application
        bodies_per_env = self.num_hand_bodies + self.num_object_bodies
        self.object_body_indices = (
            torch.arange(self.num_envs, device=self.device) * bodies_per_env
            + self.num_hand_bodies  # object body is after hand bodies
        ).to(torch.long)

        # Allocate pseudo-gravity force tensor: (total_bodies, 3)
        total_bodies = self.rb_state.shape[0]
        self._pseudo_gravity_forces = torch.zeros(
            total_bodies, 3, device=self.device
        )
        self._pseudo_gravity_torques = torch.zeros(
            total_bodies, 3, device=self.device
        )

    def _find_fingertip_indices(self) -> torch.Tensor:
        """Find rigid body indices for the 5 fingertips."""
        indices = []
        for name in FINGERTIP_NAMES:
            idx = self.gym.find_actor_rigid_body_index(
                self.envs[0], self.hand_handles[0], name, gymapi.DOMAIN_ENV
            )
            if idx < 0:
                # Fallback: use last 5 bodies
                total_bodies = self.num_hand_bodies
                indices = list(range(total_bodies - 5, total_bodies))
                break
            indices.append(idx)
        return to_torch(indices, dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _compute_observations(self) -> None:
        """Build the 118-D observation vector."""
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # Fingertip positions (relative to hand root)
        ft_pos = self._get_fingertip_positions()  # (N, 5, 3)
        ft_pos_flat = ft_pos.reshape(self.num_envs, -1)  # (N, 15)

        # Fingertip contact forces
        ft_force = self._get_fingertip_forces()  # (N, 5, 3)
        ft_force_flat = ft_force.reshape(self.num_envs, -1)  # (N, 15)

        # Object position relative to hand
        obj_pos_rel = self.object_pos - self.hand_pos  # (N, 3)

        # Convert quaternions from IsaacGym (x,y,z,w) to our (w,x,y,z) convention
        object_quat_wxyz = xyzw_to_wxyz(self.object_quat)  # (N, 4)
        target_quat_wxyz = xyzw_to_wxyz(self.target_quat)  # (N, 4)

        # Gravity vector (from DR params; supports both unified and legacy)
        gravity_vec = self.per_env_gravity  # always up-to-date from _apply_dr_params

        self.obs_buf = torch.cat([
            self.hand_dof_pos,        # 24
            self.hand_dof_vel,        # 24
            obj_pos_rel,              # 3
            object_quat_wxyz,         # 4
            self.object_linvel,       # 3
            self.object_angvel,       # 3
            target_quat_wxyz,         # 4
            ft_pos_flat,              # 15
            ft_force_flat,            # 15
            self.prev_actions,        # 20
            gravity_vec,              # 3
        ], dim=-1)

    def _get_fingertip_positions(self) -> torch.Tensor:
        """Return fingertip positions: (N, 5, 3)."""
        all_rb = self.rb_state.view(self.num_envs, -1, 13)
        ft_states = all_rb[:, self.fingertip_indices, :3]
        return ft_states

    def _get_fingertip_forces(self) -> torch.Tensor:
        """Return fingertip contact forces: (N, 5, 3)."""
        all_cf = self.contact_force.view(self.num_envs, -1, 3)
        ft_forces = all_cf[:, self.fingertip_indices, :]
        return ft_forces

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _compute_rewards(self) -> None:
        """Compute per-step reward and determine resets."""
        # Convert quaternions from IsaacGym (x,y,z,w) to our (w,x,y,z) convention
        object_quat_wxyz = xyzw_to_wxyz(self.object_quat)
        target_quat_wxyz = xyzw_to_wxyz(self.target_quat)

        # Rotation reward
        rot_reward, success = R.rotation_reward(
            object_quat_wxyz,
            target_quat_wxyz,
            rot_eps=self.task_cfg.success_tolerance,
            rot_reward_scale=self.task_cfg.rotation_reward_scale,
            success_bonus=self.task_cfg.reach_goal_bonus,
        )

        # Drop penalty
        drop_penalty, dropped = R.drop_penalty(
            self.object_pos, self.hand_pos,
            threshold=self.task_cfg.fall_dist,
            penalty=self.task_cfg.fall_penalty,
        )

        # Action penalty
        act_pen = R.action_penalty(
            self.prev_actions,
            scale=self.task_cfg.action_penalty_scale,
        )

        self.rew_buf = rot_reward + drop_penalty + act_pen

        # Track successes
        self.successes = success
        self.consecutive_successes = torch.where(
            success > 0,
            self.consecutive_successes + 1,
            torch.zeros_like(self.consecutive_successes),
        )

        # Change target on success
        if self.task_cfg.target_change_on_success:
            success_ids = (success > 0).nonzero(as_tuple=False).squeeze(-1)
            if len(success_ids) > 0:
                self._randomize_targets(success_ids)

        # Determine resets
        self.reset_buf = torch.zeros_like(self.reset_buf)
        # Timeout
        self.reset_buf = torch.where(
            self.progress_buf >= self.max_episode_length,
            torch.ones_like(self.reset_buf),
            self.reset_buf,
        )
        # Object dropped
        self.reset_buf = torch.where(
            dropped > 0,
            torch.ones_like(self.reset_buf),
            self.reset_buf,
        )

        # Extras for logging
        self.extras["success_rate"] = success.mean().item()
        self.extras["consecutive_successes"] = self.consecutive_successes.mean().item()
        self.extras["rotation_error"] = R.quat_diff_rad(
            object_quat_wxyz, target_quat_wxyz
        ).mean().item()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _apply_actions(self, actions: torch.Tensor) -> None:
        """Apply 20-D position targets through PD controller."""
        self.prev_actions = actions.clone()

        # Scale actions to DOF range
        # actions ∈ [-1, 1] -> dof targets
        targets = self.hand_dof_lower[:SHADOW_HAND_NUM_ACTUATED] + (
            (actions + 1.0) * 0.5
            * (self.hand_dof_upper[:SHADOW_HAND_NUM_ACTUATED] - self.hand_dof_lower[:SHADOW_HAND_NUM_ACTUATED])
        )

        # Set DOF position targets
        self.gym.set_dof_position_target_tensor(
            self.sim,
            gymtorch.unwrap_tensor(self._expand_targets(targets)),
        )

    def _expand_targets(self, targets_actuated: torch.Tensor) -> torch.Tensor:
        """Expand 20-D actuated targets to full 24-D DOF tensor."""
        full = self.hand_dof_pos.clone()
        full[:, :SHADOW_HAND_NUM_ACTUATED] = targets_actuated
        return full.reshape(-1)

    # ------------------------------------------------------------------
    # External forces (per-env gravity workaround)
    # ------------------------------------------------------------------

    def _apply_external_forces(self) -> None:
        """Apply pseudo-gravity forces to achieve per-env gravity."""
        self._apply_pseudo_gravity_forces()

    # ------------------------------------------------------------------
    # Resets
    # ------------------------------------------------------------------

    def _reset_envs(self, env_ids: torch.Tensor) -> None:
        """Reset specific environments."""
        n = len(env_ids)
        if n == 0:
            return

        # Reset hand DOFs to default + small noise
        noise = torch_rand_float(
            -0.2, 0.2,
            (n, self.num_hand_dofs),
            device=self.device,
        )
        default = self.hand_dof_default.unsqueeze(0).expand(n, -1)
        new_dof_pos = default + noise
        new_dof_pos = tensor_clamp(
            new_dof_pos, self.hand_dof_lower, self.hand_dof_upper
        )

        # Write DOF states
        hand_dof_idx = env_ids  # simplified indexing
        self.hand_dof_pos[env_ids] = new_dof_pos
        self.hand_dof_vel[env_ids] = 0.0

        # Reset object to above palm with random orientation
        object_pos = torch.zeros(n, 3, device=self.device)
        object_pos[:, 2] = 0.75  # above the palm

        rand_angles = torch_rand_float(
            -math.pi, math.pi, (n, 3), device=self.device
        )
        object_quat = quat_from_euler_xyz(
            rand_angles[:, 0], rand_angles[:, 1], rand_angles[:, 2]
        )

        self.root_state[self.object_indices[env_ids], 0:3] = object_pos
        self.root_state[self.object_indices[env_ids], 3:7] = object_quat
        self.root_state[self.object_indices[env_ids], 7:13] = 0.0

        # Apply DR parameters (gravity, friction, mass)
        self._apply_dr_params(env_ids)

        # Randomize target
        self._randomize_targets(env_ids)

        # Write back to simulation
        actor_indices = torch.cat([
            self.hand_indices[env_ids],
            self.object_indices[env_ids],
        ]).to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state),
            gymtorch.unwrap_tensor(actor_indices),
            len(actor_indices),
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(self.hand_dof_indices[env_ids].to(torch.int32)),
            n,
        )

        self.progress_buf[env_ids] = 0
        self.consecutive_successes[env_ids] = 0
        self.prev_actions[env_ids] = 0

    def _randomize_targets(self, env_ids: torch.Tensor) -> None:
        """Sample random target orientations."""
        n = len(env_ids)
        rand_angles = torch_rand_float(
            -math.pi, math.pi, (n, 3), device=self.device
        )
        self.target_quat[env_ids] = quat_from_euler_xyz(
            rand_angles[:, 0], rand_angles[:, 1], rand_angles[:, 2]
        )

    def _apply_dr_params(self, env_ids: torch.Tensor) -> None:
        """Apply domain randomization parameters to the simulation.

        Gravity variation is implemented as per-env external forces on
        the object (since IsaacGym gravity is a global sim param).
        Object mass and friction are set per-actor as usual.
        """
        if not self.dr_params:
            return

        # --- Gravity (per-env via external force workaround) ---
        gravity_enabled = (
            self.dr_manager.axes["gravity"].enabled
            or self.dr_manager.axes["gravity_dir"].enabled
            or self.dr_manager.axes["gravity_mag"].enabled
        )
        if gravity_enabled:
            grav_vecs = self.dr_manager.get_gravity_vectors(self.dr_params)
            self.per_env_gravity[env_ids] = grav_vecs[
                :len(env_ids)] if len(grav_vecs) == len(env_ids) else grav_vecs[env_ids]

        # --- Object mass ---
        if self.dr_manager.axes["object_mass"].enabled:
            masses = self.dr_params["object_mass"]  # (N, 1)
            for idx in env_ids:
                i = idx.item()
                body_props = self.gym.get_actor_rigid_body_properties(
                    self.envs[i], self.object_handles[i]
                )
                body_props[0].mass = masses[i, 0].item()
                self.gym.set_actor_rigid_body_properties(
                    self.envs[i], self.object_handles[i], body_props
                )
                self.per_env_object_mass[i] = masses[i, 0]

        # --- Friction ---
        if self.dr_manager.axes["friction"].enabled:
            frictions = self.dr_params["friction"]
            for idx in env_ids:
                i = idx.item()
                shape_props = self.gym.get_actor_rigid_shape_properties(
                    self.envs[i], self.object_handles[i]
                )
                for sp in shape_props:
                    sp.friction = frictions[i, 0].item()
                self.gym.set_actor_rigid_shape_properties(
                    self.envs[i], self.object_handles[i], shape_props
                )

        # --- PD gains (kp, kd) ---
        if self.dr_manager.axes["kp"].enabled or self.dr_manager.axes["kd"].enabled:
            kp_scales = self.dr_params.get("kp", torch.ones(self.num_envs, 1, device=self.device))
            kd_scales = self.dr_params.get("kd", torch.ones(self.num_envs, 1, device=self.device))
            for idx in env_ids:
                i = idx.item()
                dof_props = self.gym.get_actor_dof_properties(
                    self.envs[i], self.hand_handles[i]
                )
                for d in range(self.num_hand_dofs):
                    dof_props["stiffness"][d] = self.hand_kp[d] * kp_scales[i, 0].item()
                    dof_props["damping"][d] = self.hand_kd[d] * kd_scales[i, 0].item()
                self.gym.set_actor_dof_properties(
                    self.envs[i], self.hand_handles[i], dof_props
                )

    def _apply_pseudo_gravity_forces(self) -> None:
        """Apply per-env gravity deviation as external force on objects.

        F_pseudo = (g_desired - g_global) * m_object
        Applied every simulation step via apply_rigid_body_force_tensors.
        """
        if self._pseudo_gravity_forces is None:
            return

        # Compute deviation from global gravity
        delta_g = self.per_env_gravity - self._global_gravity  # (N, 3)
        forces = delta_g * self.per_env_object_mass.unsqueeze(-1)  # (N, 3)

        # Write into the sparse force tensor
        self._pseudo_gravity_forces.zero_()
        self._pseudo_gravity_forces[self.object_body_indices] = forces

        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self._pseudo_gravity_forces),
            gymtorch.unwrap_tensor(self._pseudo_gravity_torques),
            gymapi.ENV_SPACE,
        )
