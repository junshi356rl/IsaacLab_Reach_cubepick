# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.assets import RigidObject
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul, quat_rotate_inverse, normalize
from .helper import compute_gripper_midpoint_dot
import math
from .....helpers.robotiq_fingertip_center_helper import get_left_right_fingertip_midpoint_pos_w, lab_tip_center_pos_w, FingertipInitCfg, get_left_right_fingertip_gap

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
    ) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w = env.scene.env_origins + des_pos_b
    if hasattr(asset_cfg, 'body_ids') and asset_cfg.body_ids is not None and asset_cfg.body_ids.start is not None:
        curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3] # type: ignore
    else:
        curr_pos_w = asset.data.body_state_w[:, [0], :3].squeeze() # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)

def position_target_asset_error(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_asset_cfg: SceneEntityCfg
    ) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    target_asset: RigidObject = env.scene[target_asset_cfg.name]
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3] # type: ignore
    target_pos_w = target_asset.data.body_state_w[:, [0], :3].squeeze() # type: ignore
    return torch.norm(curr_pos_w - target_pos_w, dim=1)

def position_target_asset_error_tanh(
    env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg, target_asset_cfg: SceneEntityCfg
    ) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    target_asset: RigidObject = env.scene[target_asset_cfg.name]
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3] # type: ignore
    target_pos_w = target_asset.data.body_state_w[:, [0], :3].squeeze() # type: ignore
    distance = torch.norm(curr_pos_w - target_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)

def asset_center_position_target_asset_error_tanh(
    env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg, asset_size: tuple, target_asset_cfg: SceneEntityCfg
    ) -> torch.Tensor:
    if len(asset_size) != 3:
        raise ValueError("asset_size must be a tuple of 3 elements (size_x, size_y, size_z)")
    asset: RigidObject = env.scene[asset_cfg.name]
    target_asset: RigidObject = env.scene[target_asset_cfg.name]
    curr_base_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3] # type: ignore
    # consider the current body orientation to compute the offset to the gripper tip
    curr_base_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7] # type: ignore
    offset_local = torch.tensor(
        [asset_size[0]/2, asset_size[1]/2, asset_size[2]/2],
        device=env.device
    ).unsqueeze(0)
    # expand offset_local to match batch size
    offset_local = offset_local.expand(env.num_envs, -1)
    # rotate offset to world
    curr_pos_w,_ = combine_frame_transforms(curr_base_pos_w, curr_base_quat_w, offset_local)
    target_pos_w = target_asset.data.body_state_w[:, [0], :3].squeeze() # type: ignore
    distance = torch.norm(curr_pos_w - target_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)

# def sensor_activated(
#     env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg
#     ) -> torch.Tensor:
#     sensor = env.scene[sensor_cfg.name]
#     # if an environment's sensor has contact through sensor.data.force_matrix_w, return 1 for that env, else 0
#     # Vectorized: sum absolute forces across trailing dims per env and compare against small threshold
#     nd = sensor.data.force_matrix_w.ndim
#     activated = (sensor.data.force_matrix_w.abs().sum(dim=tuple(range(1, nd))) > 0.1).to(env.device).float()
#     return activated

# def position_command_error_tanh_sensor_activated(
#     env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg
#     ) -> torch.Tensor:
#     pos_error = position_command_error_tanh(env, std, command_name, asset_cfg)
#     sensor_active = sensor_activated(env, sensor_cfg)
#     res = pos_error * sensor_active
#     return res

# def sensor_activated_on_target(
#     env: ManagerBasedRLEnv,
#     sensor_cfg: SceneEntityCfg,
#     target_body_name: str,
# ) -> torch.Tensor:
#     sensor = env.scene.sensors[sensor_cfg.name]
#     object_names = env.scene["target_asset"].body_names

#     if target_body_name not in object_names:
#         raise ValueError(f"Target body '{target_body_name}' not found in scene.")

#     target_idx = object_names.index(target_body_name)
#     force_matrix = sensor.data.force_matrix_w  # shape: (N, B, T)
#     if force_matrix is not None:
#         contact_with_target = force_matrix[..., target_idx]  # shape: (N, B)
#         total_contact_force_per_env = contact_with_target.sum(dim=1)  # sum over fingers
#         threshold = 0.1  # N
#         is_in_contact = (total_contact_force_per_env > threshold).float()
#     else:
#         is_in_contact = torch.zeros(env.num_envs, device=env.device)

#     return is_in_contact


def symmetric_finger_side_grasp_reward(
    env: ManagerBasedRLEnv,
    std: float,
    ideal_gap: float, 
    finger_tip_offset_left: list,   # left_inner_finger local tip
    finger_tip_offset_right: list,  # right_inner_finger local tip
    left_finger_name: str,
    right_finger_name: str,
    min_gap: float,
    max_gap: float,
    asset_cfg: SceneEntityCfg,
    target_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:

    robot: RigidObject = env.scene[asset_cfg.name]
    target_asset: RigidObject = env.scene[target_asset_cfg.name]

    left_tip_w = get_tip_pos(env, robot, left_finger_name, finger_tip_offset_left)
    right_tip_w = get_tip_pos(env, robot, right_finger_name, finger_tip_offset_right)
    target_asset_pos_w = target_asset.data.body_state_w[:, 0, :3]

    vec_left_to_target_asset = target_asset_pos_w - left_tip_w    # L: from left to target_asset
    vec_right_to_target_asset = target_asset_pos_w - right_tip_w  # R: from right to target_asset

    norm_L = torch.nn.functional.normalize(vec_left_to_target_asset, dim=-1)
    norm_R = torch.nn.functional.normalize(vec_right_to_target_asset, dim=-1)
    cosine_similarity = torch.sum(norm_L * norm_R, dim=-1)
    alignment_error = cosine_similarity + 1.0
    reward_align = 1 - torch.tanh(alignment_error / std)
    valid_side_contact = (cosine_similarity < 0.0).float()

    current_gap = torch.norm(left_tip_w - right_tip_w, dim=1)
    gap_deviation = torch.abs(current_gap - ideal_gap)
    reward_gap = 1 - torch.tanh(gap_deviation / std)

    in_range = ((current_gap >= min_gap) & (current_gap <= max_gap)).float()

    total_reward = (
        0.5 * reward_align +
        0.3 * reward_gap
    ) * valid_side_contact * in_range

    return total_reward

def two_finger_midpoint_to_target_asset_distance_reward(
    env: ManagerBasedRLEnv,
    std: float,
    finger_tip_offset_left: list,
    finger_tip_offset_right: list,
    left_finger_name: str,
    right_finger_name: str,
    asset_cfg: SceneEntityCfg,
    target_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot: RigidObject = env.scene[asset_cfg.name]
    target_asset: RigidObject = env.scene[target_asset_cfg.name]

    left_tip_w = get_tip_pos(env, robot, left_finger_name, finger_tip_offset_left)
    right_tip_w = get_tip_pos(env, robot, right_finger_name, finger_tip_offset_right)
    target_asset_pos_w = target_asset.data.body_state_w[:, 0, :3]

    midpoint_w = (left_tip_w + right_tip_w) / 2  # (N, 3)

    distance = torch.norm(midpoint_w - target_asset_pos_w, dim=1)

    return 1 - torch.tanh(distance / std)


def get_tip_pos(env, robot, body_name, offset):
    body_idx = robot.find_bodies(body_name)[0]
    pos_w = robot.data.body_state_w[:, body_idx, :3].squeeze(1)
    quat_w = robot.data.body_state_w[:, body_idx, 3:7].squeeze(1)
    local_offset = torch.tensor([offset], device=env.device).expand(env.num_envs, -1)
    tip_w, _ = combine_frame_transforms(pos_w, quat_w, local_offset)
    return tip_w


def gripper_fingertip_midpoint_distance(env, target_asset_pos_w: torch.Tensor) -> torch.Tensor:
    """Compute the distance from the gripper fingertip midpoint to a target position.

    Args:
        env: environment instance
        target_asset_pos_w: tensor of shape (N, 3) with target positions in world frame

    Returns:
        Tensor of shape (N,) containing the Euclidean distance per env.
    """
    gripper_fingertip_midpoint = get_left_right_fingertip_midpoint_pos_w(env)
    to_target = target_asset_pos_w - gripper_fingertip_midpoint
    distance = torch.norm(to_target, dim=1)
    return distance

def align_gripper_approach_direction_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
    left_finger_cfg: "SceneEntityCfg" = None,
    right_finger_cfg: "SceneEntityCfg" = None,
    target_asset_cfg: "SceneEntityCfg" = None,
) -> torch.Tensor:
    # Use helper to compute the gripper opening vs midpoint->target dot product
    dot_product = compute_gripper_midpoint_dot(env, left_finger_cfg, right_finger_cfg, target_asset_cfg)
    # Reward: high when |dot| ≈ 0
    # reward = 1.0 - torch.tanh((dot_product ** 2) / std)
    angle_error_rad = torch.acos(torch.abs(dot_product))
    target_angle = math.pi / 2
    alignment_error = torch.abs(angle_error_rad - target_angle)
    reward = 1.0 - torch.tanh(alignment_error / std)
    return reward

def gripper_target_dist_reward(
    env: ManagerBasedRLEnv,
    std_dist: float,
    target_asset_cfg: "SceneEntityCfg",
) -> torch.Tensor:
    target_asset: RigidObject = env.scene[target_asset_cfg.name]
    target_asset_pos_w = target_asset.data.body_state_w[:, 0, :3]
    # Calculate fingertip midpoint to target distance reward
    distance = gripper_fingertip_midpoint_distance(env, target_asset_pos_w)
    dist_reward = torch.exp(-distance / std_dist)
    return dist_reward

def gripper_grasp_cube_reward(
    env: ManagerBasedRLEnv,
    std_dist: float,
    std_grasp: float,
    target_asset_cfg: "SceneEntityCfg",
    cube_length: float,
    dist_tolerance: float,
    grasp_tolerance: float,
    grasp_success_threshold: float,
    grasp_success_reward: float,
) -> torch.Tensor:
    if not hasattr(env.unwrapped, "_fingertip_center_info") and env.unwrapped.common_step_counter==0:
        # Not initialized; return dummy value. 
        # The shape should be N*3
        return torch.zeros((env.unwrapped.num_envs), dtype=torch.float32, device=env.scene.articulations[FingertipInitCfg.ROBOT_KEY].device)
    info = env.unwrapped._fingertip_center_info
    robot = env.scene.articulations[FingertipInitCfg.ROBOT_KEY]
    target_asset: RigidObject = env.scene[target_asset_cfg.name]
    target_asset_pos_w = target_asset.data.root_pos_w[:, :3]
    

    # Calculate fingertip midpoint to target distance reward
    distance = gripper_fingertip_midpoint_distance(env, target_asset_pos_w)
    dist_factor = torch.exp(-distance / std_dist)
    
    # Check whether the fingertips are grasping the cube with a symmetric grasp
    left_tip_pos_w = lab_tip_center_pos_w(
        robot,
        parent_body_id=info["left_fingertip_id"],
        tip_center_parent_t=info["left_tip_center_parent_t"],
    )
    right_tip_pos_w = lab_tip_center_pos_w(
        robot,
        parent_body_id=info["right_fingertip_id"],
        tip_center_parent_t=info["right_tip_center_parent_t"],
    )

    left_to_target = left_tip_pos_w - target_asset_pos_w
    right_to_target = right_tip_pos_w - target_asset_pos_w
    left_to_target_len = torch.norm(left_to_target, dim=1)
    right_to_target_len = torch.norm(right_to_target, dim=1)
    eps = 1e-6
    left_vec_norm = left_to_target / (left_to_target_len.unsqueeze(-1) + eps)
    right_vec_norm = right_to_target / (right_to_target_len.unsqueeze(-1) + eps)

    # |left_vec| == |right_vec|
    length_diff = torch.abs(left_to_target_len - right_to_target_len)
    length_reward = torch.exp(-length_diff / std_grasp)
    # left_vec · right_vec == -1
    dot_product = torch.sum(left_vec_norm * right_vec_norm, dim=1)
    opposite_error = torch.abs(dot_product + 1.0)
    opposite_reward = torch.exp(-opposite_error*2)
    # |left_pos - right_pos| == target_size
    finger_gap = get_left_right_fingertip_gap(env).squeeze()
    gap_excess = torch.clamp(finger_gap - cube_length, min=0.0)
    gap_reward = torch.exp(-gap_excess / std_grasp)
    
    grasp_quality = length_reward * opposite_reward * gap_reward * dist_factor
    is_success = (
        (grasp_quality > grasp_success_threshold) &
        (distance < dist_tolerance) &
        (finger_gap < cube_length + grasp_tolerance)
    ).float()

    return grasp_quality * 2.0 + is_success * grasp_success_reward

def contact_grasp_reward(env: ManagerBasedRLEnv, target_asset_cfg: "SceneEntityCfg", force_scale: float, force_threshold: float, dist_tolerance: float) -> torch.Tensor:
    contact_data = env.scene["finger_contact_sensor"].data.net_forces_w_history
    contact_force = torch.norm(contact_data[:, -1, :], dim=-1).squeeze()
    contact_reward = torch.clamp(contact_force / force_scale, max=1.0)

    target_asset: RigidObject = env.scene[target_asset_cfg.name]
    target_asset_pos_w = target_asset.data.root_pos_w[:, :3]
    distance = gripper_fingertip_midpoint_distance(env, target_asset_pos_w)
    is_success = ((distance < dist_tolerance) &
                  (contact_force > force_threshold)).float()
    return contact_reward * is_success