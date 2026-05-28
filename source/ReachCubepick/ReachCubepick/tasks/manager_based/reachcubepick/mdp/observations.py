# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject
from .helper import (
    get_finger_midpoint_target_axis_product,
    get_cube_velocity_alignment,
    get_finger_to_target,
    get_wrist_normal_to_target,
    get_finger_line_horizontal_info,
    get_offset_body_pos_w,
    PAD_OFFSET,
    get_finger_features,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def target_position_error(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Position error vector from asset to target in world frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    target_asset: RigidObject = env.scene[target_cfg.name]

    if hasattr(asset_cfg, "body_ids") and asset_cfg.body_ids is not None:
        curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]
    else:
        curr_pos_w = asset.data.body_state_w[:, 0, :3]

    target_pos_w = target_asset.data.body_state_w[:, 0, :3]
    return target_pos_w - curr_pos_w


def target_orientation_quat(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Quaternion of the target link."""
    asset: RigidObject = env.scene[asset_cfg.name]
    if not hasattr(asset_cfg, "body_ids") or asset_cfg.body_ids is None:
        raise ValueError("asset_cfg must have body_ids defined")
    return asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]


def contact_forces(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Flattened contact forces from the sensor."""
    sensor = env.scene[sensor_cfg.name]
    return sensor.data.force_matrix_w.reshape(env.num_envs, -1)


def gripper_to_target_rad(
    env: ManagerBasedRLEnv,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Angle between gripper midpoint and target direction (radians)."""
    dot_product = get_finger_midpoint_target_axis_product(env, left_finger_cfg, right_finger_cfg, target_cfg)
    angle_rad = torch.acos(dot_product.abs().clamp(-1.0, 1.0))
    return angle_rad.unsqueeze(-1)


def fingertip_midpoint_to_target(
    env: ManagerBasedRLEnv,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Vector from fingertip midpoint to target."""
    _, _, midpoint_pos_w = get_finger_features(env, left_finger_cfg, right_finger_cfg)
    target_asset: RigidObject = env.scene[target_cfg.name]
    target_pos_w = target_asset.data.body_state_w[:, 0, :3]
    return target_pos_w - midpoint_pos_w


def finger_gap_minus_cube_length(
    env: ManagerBasedRLEnv,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    cube_length: float,
) -> torch.Tensor:
    """Difference between finger gap and cube length."""
    left_asset = env.scene[left_finger_cfg.name]
    right_asset = env.scene[right_finger_cfg.name]

    left_tip_pos_w = get_offset_body_pos_w(left_asset, left_finger_cfg.body_names[0], PAD_OFFSET)
    right_tip_pos_w = get_offset_body_pos_w(right_asset, right_finger_cfg.body_names[0], PAD_OFFSET)

    return torch.norm(left_tip_pos_w - right_tip_pos_w, dim=-1, keepdim=True) - cube_length


def finger_midpoint_and_target(
    env: ManagerBasedRLEnv,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Concatenated finger midpoint position and vector to target."""
    mid_point, to_target = get_finger_to_target(env, left_finger_cfg, right_finger_cfg, target_cfg)
    return torch.cat((mid_point, to_target), dim=-1)


def finger_quat(
    env: ManagerBasedRLEnv,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Quaternions of left and right fingers."""
    left_asset = env.scene[left_finger_cfg.name]
    right_asset = env.scene[right_finger_cfg.name]

    left_quat_w = left_asset.data.body_quat_w[:, left_finger_cfg.body_ids[0], :]
    right_quat_w = right_asset.data.body_quat_w[:, right_finger_cfg.body_ids[0], :]
    return torch.cat((left_quat_w, right_quat_w), dim=-1)


def asset_to_command_delta(
    env: ManagerBasedRLEnv,
    target_cfg: SceneEntityCfg,
    command_name: str,
) -> torch.Tensor:
    """Position delta from asset to commanded target position."""
    command = env.command_manager.get_command(command_name)
    des_pos_w = env.scene.env_origins + command[:, :3]
    curr_pos_w = env.scene[target_cfg.name].data.root_pos_w[:, :3]
    return des_pos_w - curr_pos_w


def asset_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Root linear velocity of the asset."""
    return env.scene[asset_cfg.name].data.root_vel_w


def body_vel(
    env: ManagerBasedRLEnv,
    body_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Linear velocity of a specific body."""
    body_idx = env.scene[body_cfg.name].find_bodies(body_cfg.body_names[0])[0][0]
    return env.scene[body_cfg.name].data.body_vel_w[:, body_idx, :3]


def cube_velocity_alignment(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
) -> torch.Tensor:
    """Velocity alignment score between cube and command."""
    alignment_score, *_ = get_cube_velocity_alignment(env, asset_cfg, command_name)
    return alignment_score


def wrist_normal_to_target_rad(
    env: ManagerBasedRLEnv,
    ee_link_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Angle between wrist normal and target direction (radians)."""
    dot_product = get_wrist_normal_to_target(env, ee_link_cfg, target_cfg)
    angle_rad = torch.acos(dot_product.clamp(-1.0, 1.0))
    return angle_rad.unsqueeze(-1)


def finger_line_horizontal_rad(
    env: ManagerBasedRLEnv,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Angle of the finger line relative to horizontal (radians)."""
    info = get_finger_line_horizontal_info(env, left_finger_cfg, right_finger_cfg)
    return info["angle_rad"].unsqueeze(-1)

def env_origin(env):
    res = env.scene.env_origins
    return res