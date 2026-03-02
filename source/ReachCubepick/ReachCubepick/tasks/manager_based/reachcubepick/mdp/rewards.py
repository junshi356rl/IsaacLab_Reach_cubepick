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
from .debug_helper import debug_robot_state
import math
from .....helpers.robotiq_fingertip_center_helper import get_left_right_fingertip_midpoint_pos_w, lab_tip_center_pos_w, FingertipInitCfg, get_left_right_fingertip_gap

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

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
    target_asset_pos_w = target_asset.data.root_pos_w[:, :3]
    # Calculate fingertip midpoint to target distance reward
    distance = gripper_fingertip_midpoint_distance(env, target_asset_pos_w)
    dist_reward = torch.exp(-distance / std_dist) # not work
    # dist_reward = 1.0 - torch.tanh(distance / std_dist)
    if env.unwrapped.common_step_counter % 1000 == 0:
        print(f"[DEBUG] Step {env.unwrapped.common_step_counter}:")
        print(f"  distance - mean: {distance.mean().item():.4f}m, "
              f"max: {distance.max().item():.4f}m")
        print(f"  dist_reward - mean: {dist_reward.mean().item():.4f}, "
              f"max: {dist_reward.max().item():.4f}")
        debug_robot_state(env)
    return dist_reward

def gripper_grasp_cube_reward(
    env: ManagerBasedRLEnv,
    std_dist: float,
    std_grasp: float,
    target_asset_cfg: "SceneEntityCfg",
    dist_tolerance: float,
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
    opposite_reward = torch.exp(-opposite_error*3.0)
    
    grasp_quality = length_reward * opposite_reward * dist_factor
    is_success = (
        (grasp_quality > grasp_success_threshold) &
        (distance < dist_tolerance)
    ).float()

    if env.unwrapped.common_step_counter % 10000 == 0 and env.unwrapped.common_step_counter > 0:
        print(f"[DEBUG] Step {env.unwrapped.common_step_counter}:")
        print(f"  dot_product - mean: {dot_product.mean().item():.4f}, "
              f"min: {dot_product.min().item():.4f}, max: {dot_product.max().item():.4f}")
        print(f"  opposite_reward - mean: {opposite_reward.mean().item():.4f}, "
              f"non-zero: {(opposite_reward > 0.01).sum().item()}/{env.num_envs}")
        print(f"  grasp_quality - mean: {grasp_quality.mean().item():.4f}")
        print(f"  is_success - count: {is_success.sum().item()}/{env.num_envs}")

    return grasp_quality * 2.0 + is_success * grasp_success_reward

# def contact_grasp_reward(env: ManagerBasedRLEnv, force_scale: float) -> torch.Tensor:
#     contact_data = env.scene["finger_contact_sensor"].data.net_forces_w_history
#     contact_force = torch.norm(contact_data[:, -1, :], dim=-1).squeeze()
#     contact_reward = torch.clamp(contact_force / force_scale, max=1.0)

#     if env.unwrapped.common_step_counter % 10000 == 0 and env.unwrapped.common_step_counter > 0:
#         has_contact = contact_force > 0.1
#         num_contact = has_contact.sum().item()
#         print(f"[DEBUG] Step {env.unwrapped.common_step_counter}:")
#         print(f"  Contact - envs: {num_contact}/{env.num_envs} ({100*num_contact/env.num_envs:.1f}%), "
#               f"force: {contact_force[has_contact].mean().item() if num_contact > 0 else 0:.4f}N, "
#               f"reward: {contact_reward.mean().item():.6f}")

#     return contact_reward 

def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
    ) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w = env.scene.env_origins + des_pos_b
    curr_pos_w = asset.data.root_pos_w[:, :3]
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)

def finger_gap_reward(
    env: ManagerBasedRLEnv,
    cube_length: float = 0.08,
    gap_far_offset: float = 0.06,    
    gap_near_offset: float = -0.005, 
    gap_std: float = 0.02,
    dist_far: float = 0.20,          
    dist_near: float = 0.05,         
) -> torch.Tensor:
    finger_gap = get_left_right_fingertip_gap(env).squeeze()
    
    target_asset: RigidObject = env.scene["cube"]
    target_pos_w = target_asset.data.root_pos_w[:, :3]
    gripper_midpoint = get_left_right_fingertip_midpoint_pos_w(env)
    distance = torch.norm(target_pos_w - gripper_midpoint, dim=1)
    
    gap_far = cube_length + gap_far_offset
    gap_near = cube_length + gap_near_offset
    
    t = (distance - dist_near) / (dist_far - dist_near)
    t = torch.clamp(t, 0.0, 1.0) 
    
    target_gap_interp = gap_near + t * (gap_far - gap_near)
    
    gap_error = torch.abs(finger_gap - target_gap_interp)
    reward = torch.exp(-gap_error / gap_std)
    
    return reward

def get_finger_features(env: ManagerBasedRLEnv, left_finger_cfg, right_finger_cfg):
    left_finger_asset_idx = env.scene[left_finger_cfg.name].find_bodies(left_finger_cfg.body_names[0])[0][0]
    left_finger_pos_w = env.scene[left_finger_cfg.name].data.body_pos_w[:, left_finger_asset_idx, :3]
    right_finger_asset_idx = env.scene[right_finger_cfg.name].find_bodies(right_finger_cfg.body_names[0])[0][0]
    right_finger_pos_w = env.scene[right_finger_cfg.name].data.body_pos_w[:, right_finger_asset_idx, :3]
    midpoint_pos_w = (left_finger_pos_w + right_finger_pos_w) / 2.0
    return left_finger_pos_w, right_finger_pos_w, midpoint_pos_w

def get_target_distance(env, target_asset_cfg, midpoint_pos_w):
    target_asset: RigidObject = env.scene[target_asset_cfg.name]
    target_asset_pos_w = target_asset.data.root_pos_w[:, :3]
    to_target = target_asset_pos_w - midpoint_pos_w
    distance = torch.norm(to_target, dim=1)
    return distance

def native_finger_midpoint_to_target_distance_reward(env: ManagerBasedRLEnv, std_dist, left_finger_cfg, right_finger_cfg, target_asset_cfg) -> torch.Tensor:
    _, _, midpoint_pos_w = get_finger_features(env, left_finger_cfg, right_finger_cfg)
    distance = get_target_distance(env, target_asset_cfg, midpoint_pos_w)
    dist_reward = 1.0 - torch.tanh(distance / std_dist)
    if env.unwrapped.common_step_counter % 1000 == 0:
        print(f"[DEBUG] Step {env.unwrapped.common_step_counter}:")
        print(f"  distance - mean: {distance.mean().item():.4f}m, "
              f"max: {distance.max().item():.4f}m")
        print(f"  dist_reward - mean: {dist_reward.mean().item():.4f}, "
              f"max: {dist_reward.max().item():.4f}")
        debug_robot_state(env)
    return dist_reward

def native_finger_grasp_reward(env: ManagerBasedRLEnv, std_dist, left_finger_cfg, right_finger_cfg, target_asset_cfg) -> torch.Tensor:
    target_asset: RigidObject = env.scene[target_asset_cfg.name]
    target_asset_pos_w = target_asset.data.root_pos_w[:, :3]
    left_finger_pos_w, right_finger_pos_w, midpoint_pos_w = get_finger_features(env, left_finger_cfg, right_finger_cfg)
    
    distance = get_target_distance(env, target_asset_cfg, midpoint_pos_w)
    dist_factor = torch.exp(-distance / std_dist)
    
    left_to_target = left_finger_pos_w - target_asset_pos_w
    right_to_target = right_finger_pos_w - target_asset_pos_w
    left_to_target_len = torch.norm(left_to_target, dim=1)
    right_to_target_len = torch.norm(right_to_target, dim=1)
    eps = 1e-6
    left_vec_norm = left_to_target / (left_to_target_len.unsqueeze(-1) + eps)
    right_vec_norm = right_to_target / (right_to_target_len.unsqueeze(-1) + eps)

    length_diff = torch.abs(left_to_target_len - right_to_target_len)
    length_reward = torch.exp(-length_diff / std_dist)
    
    dot_product = torch.sum(left_vec_norm * right_vec_norm, dim=1)
    opposite_error = torch.abs(dot_product + 1.0)
    opposite_reward = torch.exp(-opposite_error*3.0)
    
    grasp_quality = length_reward * opposite_reward * dist_factor

    return grasp_quality

def native_finger_gap_reward(
        env: ManagerBasedRLEnv,
        left_finger_cfg,
        right_finger_cfg,
        target_asset_cfg,
        cube_length,
        std_dist,         
    ):
    left_finger_pos_w, right_finger_pos_w, midpoint_pos_w = get_finger_features(env, left_finger_cfg, right_finger_cfg)
    distance = get_target_distance(env, target_asset_cfg, midpoint_pos_w)
    finger_gap = torch.norm(left_finger_pos_w - right_finger_pos_w, dim=1)
    dist_factor = torch.exp(-distance / std_dist)
    
    return torch.exp(-torch.abs(finger_gap - cube_length) / std_dist) * dist_factor