# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-path timeopt before TrajOpt result selection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import torch

from curobo._src.motion.motion_trajectory_timeopt import PathSpeedBound, TrajectoryTimeOpt
from curobo._src.motion.motion_trajectory_timeopt_cfg import TrajectoryTimeOptCfg
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.rollout.metrics import RolloutMetrics
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_joint_ops import repeat_joint_state_seeds
from curobo._src.types.control_space import ControlSpace
from curobo._src.util.logging import log_and_raise

if TYPE_CHECKING:
    from curobo._src.solver.solver_trajopt import TrajOptSolver


def trajectory_validation_masks(
    metrics: RolloutMetrics, dynamics_valid: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    """Separate per-lane pose, collision, and joint-limit/dynamic acceptance for diagnostics."""
    position = metrics.state.joint_state.position
    lanes = position.shape[0]
    path = torch.ones(lanes, device=position.device, dtype=torch.bool)
    collision = torch.ones_like(path)
    dynamic = torch.ones_like(path) if dynamics_valid is None else dynamics_valid.clone()
    constraints = metrics.costs_and_constraints
    for collection in (constraints.constraints, constraints.hybrid_costs_constraints):
        for name, values in zip(collection.names, collection.values):
            valid = (values <= 0).reshape(lanes, -1).all(dim=-1)
            if "collision" in name:
                collision = collision & valid
            elif "pose" in name or "relative_transform" in name:
                path = path & valid
            else:
                dynamic = dynamic & valid
    for name, values in zip(metrics.convergence.names, metrics.convergence.values):
        if name == "tool_pose_path_valid":
            path = path & values.reshape(lanes, -1).all(dim=-1)
    return {"path_valid": path, "collision_valid": collision, "limits_and_dynamics_valid": dynamic}


class TimeOptSolver:
    """Optimize timing and validate all retained seeds without changing their geometry."""

    tool_speed_margin = 0.99
    """Fraction of each tool speed limit timeopt targets; the grid is sampled, validation dense."""

    def __init__(self, solver: TrajOptSolver, lanes: int) -> None:
        """Allocate fixed-shape timeopt and executable-trajectory buffers."""
        if solver.transition_model.control_space != ControlSpace.BSPLINE_3:
            log_and_raise("TrajOpt timeopt requires cubic B-spline trajectories")
        self.config = solver.config
        self._set_reference_samples = solver._set_dense_reference_samples
        self.rollout = solver.additional_metrics_rollouts["interpolated_rollout"]
        self.limits = solver.kinematics.config.kinematics_config.joint_limits
        self.lanes = lanes
        dynamics = solver.transition_model.robot_dynamics
        gravity = tuple(dynamics.config.gravity) if dynamics is not None else (0.0, 0.0, -9.81)
        self.optimizer = TrajectoryTimeOpt(
            TrajectoryTimeOptCfg(
                robot=solver.kinematics.config,
                batch_size=lanes,
                n_knots=solver.action_horizon,
                gravity=gravity,
                use_cuda_graph=solver.config.use_cuda_graph,
                enable_inverse_dynamics=solver.config.enable_inverse_dynamics,
            )
        )
        capacity = solver.config.interpolation_buffer_size
        if capacity < 2:
            log_and_raise("TimeOpt requires interpolation_buffer_size >= 2")
        self._sample_index = torch.arange(capacity, **solver.device_cfg.as_torch_dict())[None]
        self._sample_dt = torch.zeros(lanes, **solver.device_cfg.as_torch_dict())
        self._trajopt = solver
        # Bound timing by the same tool speed criteria the validation rollout checks.
        self._tool_speed_costs = [
            manager.get_cost("tool_velocity_limit")
            for manager in (
                self.rollout.metrics_constraint_manager,
                self.rollout.metrics_hybrid_cost_constraint_manager,
                self.rollout.metrics_cost_manager,
            )
            if manager is not None and manager.has_cost("tool_velocity_limit")
        ]
        self._kinematics = (
            Kinematics(solver.kinematics.config, compute_jacobian=True, compute_spheres=False)
            if self._tool_speed_costs
            else None
        )
        self._tool_bound = self.tool_speed_bound
        self._bound_key: tuple = (None, ())

    def tool_speed_bound(self, path: torch.Tensor) -> torch.Tensor:
        """Largest path speed [B,G] keeping every enabled projected tool speed within its limit."""
        q, dq = path[0], path[1]
        jacobians = self._kinematics.compute_kinematics(JointState(position=q)).tool_jacobians
        twist = (jacobians @ dq[:, :, None, :, None])[..., 0]  # Tool twist per unit ds/dt.
        bound = torch.full_like(q[..., 0], torch.inf)
        for cost in self._tool_speed_costs:
            if not cost.enabled:
                continue
            criteria = cost._stacked_tool_velocity_limit_criteria
            for part, kind in ((twist[..., :3], "linear"), (twist[..., 3:], "angular")):
                projector = getattr(criteria, kind + "_projector")
                scale = getattr(criteria, kind + "_axis_scale")
                limit = getattr(criteria, "max_" + kind + "_speed")
                speed = (scale * (projector @ part[..., None])[..., 0]).norm(dim=-1)
                link = torch.where(
                    speed > 0, self.tool_speed_margin * limit / speed, torch.inf
                )
                bound = torch.minimum(bound, link.amin(dim=-1))
        return bound

    def _set_path_speed_bound(self) -> None:
        """Combine the TrajOpt solver's custom bound with enabled tool speed criteria."""
        custom = self._trajopt.path_speed_bound
        enabled = tuple(cost.enabled for cost in self._tool_speed_costs)
        if (custom, enabled) == self._bound_key:
            return
        self._bound_key = (custom, enabled)
        bounds = [b for b in (custom, self._tool_bound if any(enabled) else None) if b]
        bound: Optional[PathSpeedBound] = bounds[0] if bounds else None
        if len(bounds) == 2:

            def bound(path: torch.Tensor) -> torch.Tensor:
                return torch.minimum(bounds[0](path), bounds[1](path))

        self.optimizer.set_path_speed_bound(bound)

    def timeopt(
        self, native: JointState, current_state: JointState, num_seeds: int
    ) -> tuple[JointState, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return dense states, sample counts, feasibility, and durations for every seed.

        Counts include the exact endpoint. Each trajectory has uniform dt no larger
        than the requested interpolation_dt. Buffer overflow rejects a seed.
        """
        self._set_path_speed_bound()
        start = repeat_joint_state_seeds(current_state, num_seeds)
        timeopt = self.optimizer.timeopt(
            native.knot.contiguous(),
            start,
            native.position[:, -1].contiguous(),
            knot_dt=native.knot_dt.contiguous(),
        )
        duration = timeopt.duration
        capacity = self._sample_index.shape[1]
        intervals = torch.ceil(duration / self.config.interpolation_dt).clamp(min=1)
        fits = torch.isfinite(duration) & (intervals < capacity)
        intervals = torch.nan_to_num(intervals, nan=1, posinf=capacity - 1).clamp(max=capacity - 1)
        self._sample_dt.copy_(duration / intervals)
        times = self._sample_dt[:, None] * self._sample_index
        sampled = self.optimizer.sample(times.contiguous())
        self._set_reference_samples(
            self.optimizer._sample_progress, intervals.to(dtype=torch.int32) + 1
        )
        sampled.dt = self._sample_dt
        finite = torch.ones_like(timeopt.success)
        for name in ("position", "velocity", "acceleration", "jerk"):
            values = getattr(sampled, name)
            finite &= torch.isfinite(values).all(dim=(-2, -1))
        limits = self.limits
        for name in ("velocity", "acceleration", "jerk"):
            values = getattr(sampled, name)
            bounds = getattr(limits, name)
            finite &= ((values >= bounds[0] - 1e-4) & (values <= bounds[1] + 1e-4)).all(
                dim=(-2, -1)
            )
        state = self.rollout.transition_model.compute_augmented_state(sampled)
        if self.config.enable_inverse_dynamics:
            torque = state.joint_torque
            if torque is None:
                torque = self.optimizer._compute_torque(sampled)
            finite &= torch.isfinite(torque).all(dim=(-2, -1))
            finite &= ((torque >= limits.effort[0] - 1e-4) & (torque <= limits.effort[1] + 1e-4)).all(
                dim=(-2, -1)
            )
        metrics = self.rollout.compute_metrics_from_state(state)
        feasible = metrics.feasible
        if isinstance(feasible, torch.Tensor):
            feasible = feasible.reshape(self.lanes, -1).all(dim=-1)
        success = timeopt.success & fits & finite & feasible
        self.validation_masks = trajectory_validation_masks(
            metrics,
            timeopt.success & fits & finite,
        )
        return sampled, intervals.to(torch.int64) + 1, success, duration

    def reset_cuda_graph(self) -> None:
        """Release captured timeopt and sampling work."""
        self.optimizer.reset_cuda_graph()
