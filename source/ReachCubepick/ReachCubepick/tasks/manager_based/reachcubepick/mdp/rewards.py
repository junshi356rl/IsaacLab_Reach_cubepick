# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import math
import torch
from typing import TYPE_CHECKING, Optional
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject
from .helper import (
    get_cube_velocity_alignment,
    get_wrist_normal_to_target,
    get_finger_line_horizontal_info,
    get_finger_features,
)
from .debug_helper import debug_robot_state, debug_robot_joint_acc

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def log_step(env: ManagerBasedRLEnv, freq: int, tag: str, **metrics) -> None:
    step = env.unwrapped.common_step_counter
    if step == 0 or step % freq != 0:
        return

    parts = [f"[{tag}] Step {step}"]
    for name, val in metrics.items():
        if isinstance(val, torch.Tensor):
            val = val.detach()
            parts.append(f"{name}: mean={val.mean().item():.4f}, max={val.max().item():.4f}")
        elif isinstance(val, (int, float)):
            parts.append(f"{name}: {val:.4f}")
        else:
            parts.append(f"{name}: {val}")
    print(" | ".join(parts))


def contact_grasp_reward(
    env: ManagerBasedRLEnv,
    force_scale: float,
    sensor1_cfg: SceneEntityCfg,
    sensor2_cfg: Optional[SceneEntityCfg] = None,
    print_freq: int = 10000,
    contact_threshold: float = 0.1,
) -> torch.Tensor:
    s1 = env.scene[sensor1_cfg.name]
    f1 = torch.norm(s1.data.force_matrix_w.squeeze(dim=(1, 2)), dim=-1)

    if sensor2_cfg is not None:
        s2 = env.scene[sensor2_cfg.name]
        f2 = torch.norm(s2.data.force_matrix_w.squeeze(dim=(1, 2)), dim=-1)
    else:
        f2 = torch.zeros_like(f1)

    bilateral_force = torch.minimum(f1, f2)
    contact_reward = torch.tanh(bilateral_force / force_scale)

    log_step(
        env, print_freq, "ContactGrasp",
        f1_mean=f1[f1 > contact_threshold].mean().item() if (f1 > contact_threshold).any() else 0.0,
        f2_mean=f2[f2 > contact_threshold].mean().item() if (f2 > contact_threshold).any() else 0.0,
        reward=contact_reward,
    )
    return contact_reward


def get_target_distance(
    env: ManagerBasedRLEnv,
    target_asset_cfg: SceneEntityCfg,
    midpoint_pos_w: torch.Tensor,
) -> torch.Tensor:
    target_asset: RigidObject = env.scene[target_asset_cfg.name]
    target_pos_w = target_asset.data.root_pos_w[:, :3]
    to_target = target_pos_w - midpoint_pos_w
    return torch.norm(to_target, dim=-1)


def native_finger_midpoint_to_target_reward(
    env: ManagerBasedRLEnv,
    std_dist: float,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    target_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    _, _, midpoint_pos_w = get_finger_features(env, left_finger_cfg, right_finger_cfg)
    distance = get_target_distance(env, target_asset_cfg, midpoint_pos_w)
    dist_reward = torch.exp(-distance / std_dist)

    debug_robot_state(env)
    debug_robot_joint_acc(env, print_freq_stats=10000)
    log_step(env, 1000, "MidpointDist", distance=distance, reward=dist_reward)
    return dist_reward


def finger_closure_reward(
    env: ManagerBasedRLEnv,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    target_width: float,
    activation_dist: float,
    std_gap: float,
    gap_closure_factor: float = 0.99,
) -> torch.Tensor:
    left_pos_w, right_pos_w, midpoint_pos_w = get_finger_features(env, left_finger_cfg, right_finger_cfg)
    gap = torch.norm(left_pos_w - right_pos_w, dim=-1)
    target_gap = target_width * gap_closure_factor
    gap_error = torch.abs(gap - target_gap)

    cube_pos_w = env.scene["cube"].data.root_pos_w[:, :3]
    dist = torch.norm(midpoint_pos_w - cube_pos_w, dim=-1)

    closure_reward = torch.clamp(1.0 - gap_error / std_gap, min=0.0, max=1.0)
    reward = torch.where(dist <= activation_dist, closure_reward, torch.zeros_like(closure_reward))

    log_step(env, 1000, "ClosureReward", dist=dist, gap=gap, gap_error=gap_error, reward=reward)
    return reward


def finger_height_alignment_reward(
    env: ManagerBasedRLEnv,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    target_asset_cfg: SceneEntityCfg,
    std_height: float = 0.02,
) -> torch.Tensor:
    """Reward when fingers are at similar height as cube center (side grasp)."""
    target_asset = env.scene[target_asset_cfg.name]
    cube_pos = target_asset.data.root_pos_w[:, :3]
    _, _, midpoint_pos_w = get_finger_features(env, left_finger_cfg, right_finger_cfg)

    height_diff = torch.abs(midpoint_pos_w[:, 2] - cube_pos[:, 2])
    reward = torch.exp(-height_diff / std_height)

    log_step(env, 10000, "HeightAlign", mean_diff=height_diff, reward=reward)
    return reward


def finger_symmetry_reward(
    env: ManagerBasedRLEnv,
    target_asset_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    std_grasp: float,
) -> torch.Tensor:
    target_pos = env.scene[target_asset_cfg.name].data.root_pos_w[:, :3]
    left_pos_w, right_pos_w, _ = get_finger_features(env, left_finger_cfg, right_finger_cfg)
    l_dist = torch.norm(left_pos_w - target_pos, dim=-1)
    r_dist = torch.norm(right_pos_w - target_pos, dim=-1)
    diff = torch.abs(l_dist - r_dist)
    return torch.exp(-diff / std_grasp)


def finger_opposition_reward(
    env: ManagerBasedRLEnv,
    target_asset_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    opposition_scale: float = 3.0,
) -> torch.Tensor:
    target_pos = env.scene[target_asset_cfg.name].data.root_pos_w[:, :3]
    left_pos_w, right_pos_w, _ = get_finger_features(env, left_finger_cfg, right_finger_cfg)

    v_left = left_pos_w - target_pos
    v_right = right_pos_w - target_pos
    v_left_n = v_left / (torch.norm(v_left, dim=-1, keepdim=True) + 1e-6)
    v_right_n = v_right / (torch.norm(v_right, dim=-1, keepdim=True) + 1e-6)
    dot = (v_left_n * v_right_n).sum(dim=-1)
    return torch.exp(-torch.abs(dot + 1.0) * opposition_scale)


def position_command_error_progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor1_cfg: SceneEntityCfg,
    sensor2_cfg: SceneEntityCfg,
    max_track_dist: float = 1.2,
    dist_sigma: float = 0.12,
    grasp_force_threshold: float = 5.0,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w = env.scene.env_origins + command[:, :3]
    curr_pos_w = asset.data.root_pos_w[:, :3]
    distance = torch.norm(curr_pos_w - des_pos_w, dim=-1)

    far_reward = torch.clamp(1.0 - distance / max_track_dist, min=0.0, max=1.0)
    near_reward = torch.exp(-distance / dist_sigma)
    dist_reward = far_reward + near_reward

    s1 = env.scene[sensor1_cfg.name]
    s2 = env.scene[sensor2_cfg.name]
    f1 = torch.norm(s1.data.force_matrix_w.squeeze(dim=(1, 2)), dim=-1)
    f2 = torch.norm(s2.data.force_matrix_w.squeeze(dim=(1, 2)), dim=-1)
    bilateral_force = torch.minimum(f1, f2)
    is_grasping = (bilateral_force > grasp_force_threshold).float()

    reward = dist_reward * is_grasping
    log_step(env, 1000, "MoveProgress", distance=distance, dist_reward=dist_reward, grasp_gate=is_grasping, reward=reward)
    return reward


def joint_limit_distance_clamped(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    margin: float = 0.15,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    limits = asset.data.soft_joint_pos_limits[:, :6]
    lower, upper = limits[:, :, 0], limits[:, :, 1]
    current = asset.data.joint_pos[:, :6]

    dist_to_lower = torch.clamp(margin - (current - lower), min=0.0)
    dist_to_upper = torch.clamp(margin - (upper - current), min=0.0)

    env_hit_lower = (dist_to_lower > 0).any(dim=1)
    env_hit_upper = (dist_to_upper > 0).any(dim=1)
    env_hit_any = env_hit_lower | env_hit_upper

    log_step(
        env, 1000, "JointLimit",
        lower=f"{int(env_hit_lower.sum().item())}/{env.scene.num_envs}",
        upper=f"{int(env_hit_upper.sum().item())}/{env.scene.num_envs}",
        any=f"{int(env_hit_any.sum().item())}/{env.scene.num_envs}",
    )
    return (dist_to_lower**2 + dist_to_upper**2).sum(dim=1)


def asset_vel_to_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    cube_length: float,
) -> torch.Tensor:
    alignment_score, _, _, _, _, to_target_dist = get_cube_velocity_alignment(env, asset_cfg, command_name)
    return torch.where(
        to_target_dist < cube_length / 2.0,
        torch.ones_like(alignment_score),
        alignment_score,
    ).squeeze(dim=-1)


def debug_gripper_y_axis(
    env: ManagerBasedRLEnv,
    ee_link_cfg: SceneEntityCfg,
    target_asset_cfg: SceneEntityCfg,
    good_angle: float = 30.0,
    bad_angle: float = 60.0,
) -> None:
    dot_product = get_wrist_normal_to_target(env, ee_link_cfg, target_asset_cfg)
    angle_rad = torch.acos(dot_product.clamp(-1.0, 1.0))
    angle_deg = angle_rad * 180.0 / math.pi

    log_step(
        env, 1000, "GripperYAxis",
        dot_product=dot_product,
        angle_deg=angle_deg,
        good=f"{(angle_deg < good_angle).sum().item()}/{env.num_envs}",
        bad=f"{(angle_deg > bad_angle).sum().item()}/{env.num_envs}",
    )


def wrist_outside_normal_to_target_reward(
    env: ManagerBasedRLEnv,
    ee_link_cfg: SceneEntityCfg,
    target_asset_cfg: SceneEntityCfg,
    std_angle: float = 0.5,
) -> torch.Tensor:
    dot_product = get_wrist_normal_to_target(env, ee_link_cfg, target_asset_cfg)
    reward = torch.exp(-(1.0 - dot_product) ** 2 / std_angle)

    log_step(
        env, 10000, "YAxisApproach",
        dot_mean=dot_product,
        reward_mean=reward,
        good=f"{(reward > 0.8).sum().item()}/{env.num_envs}",
    )
    debug_gripper_y_axis(env, ee_link_cfg, target_asset_cfg)
    return reward


def finger_line_horizontal_reward(
    env: ManagerBasedRLEnv,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    std_angle: float = 0.3,
    good_angle: float = 15.0,
) -> torch.Tensor:
    info = get_finger_line_horizontal_info(env, left_finger_cfg, right_finger_cfg)
    reward = torch.exp(-(info["angle_rad"] ** 2) / std_angle)

    log_step(
        env, 10000, "FingerHorizontal",
        angle_deg=info["angle_deg"],
        reward_mean=reward,
        good=f"{(info['angle_deg'] < good_angle).sum().item()}/{env.num_envs}",
    )
    return reward
