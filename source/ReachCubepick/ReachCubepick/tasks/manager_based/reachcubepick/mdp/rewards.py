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
from .helper import compute_gripper_midpoint_dot, compute_cube_velocity_alignment, wrist_outside_normal_to_target, get_finger_line_horizontal_info
from .debug_helper import debug_robot_state, debug_cube_move_state
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

def contact_grasp_reward(env: ManagerBasedRLEnv, force_scale: float, sensor1_cfg: SceneEntityCfg, sensor2_cfg: SceneEntityCfg) -> torch.Tensor:
    s1 = env.scene[sensor1_cfg.name]
    s2 = env.scene[sensor2_cfg.name]
    f1 = torch.norm(s1.data.force_matrix_w.squeeze(dim=(1,2)), dim=-1)
    f2 = torch.norm(s2.data.force_matrix_w.squeeze(dim=(1,2)), dim=-1)
    
    avg_force = (f1 + f2) / 2.0
    contact_reward = torch.tanh(avg_force / force_scale)

    if env.unwrapped.common_step_counter % 10000 == 0 and env.unwrapped.common_step_counter > 0:
        if sensor2_cfg is not None:
            has_contact = (f1 > 0.1) | (f2 > 0.1)
            num_contact = has_contact.sum().item()
            print(f"[DEBUG] Step {env.unwrapped.common_step_counter}:")
            print(f"  Contact - envs: {num_contact}/{env.num_envs} ({100*num_contact/env.num_envs:.1f}%)")
            print(f"  force1 - mean: {f1[f1>0.1].mean().item() if (f1>0.1).any() else 0:.4f}N, "
                  f"force2 - mean: {f2[f2>0.1].mean().item() if (f2>0.1).any() else 0:.4f}N")
            print(f"  reward - mean: {contact_reward.mean().item():.6f}")
        else:
            has_contact = f1 > 0.1
            num_contact = has_contact.sum().item()
            print(f"[DEBUG] Step {env.unwrapped.common_step_counter}:")
            print(f"  Contact - envs: {num_contact}/{env.num_envs} ({100*num_contact/env.num_envs:.1f}%), "
                  f"force: {f1[has_contact].mean().item() if num_contact > 0 else 0:.4f}N, "
                  f"reward: {contact_reward.mean().item():.6f}")

    return contact_reward

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

def print_reward(
        env,
        print_freq: int,
        metric_name: str,
        metric_value: torch.Tensor,
        reward_name: str,
        reward_value: torch.Tensor,
    ):
    if env.unwrapped.common_step_counter % print_freq == 0:
        print(f"[DEBUG] Step {env.unwrapped.common_step_counter} - {reward_name}:")
        print(f"  {metric_name} - mean: {metric_value.mean().item():.4f}, "
              f"max: {metric_value.max().item():.4f}")
        print(f"  reward - mean: {reward_value.mean().item():.4f}, "
              f"max: {reward_value.max().item():.4f}")

def native_finger_midpoint_to_target_distance_reward(env: ManagerBasedRLEnv, std_dist, left_finger_cfg, right_finger_cfg, target_asset_cfg) -> torch.Tensor:
    _, _, midpoint_pos_w = get_finger_features(env, left_finger_cfg, right_finger_cfg)
    distance = get_target_distance(env, target_asset_cfg, midpoint_pos_w)
    # dist_reward = 1.0 - torch.tanh(distance / std_dist)
    dist_reward = torch.exp(-distance / std_dist)
    debug_robot_state(env)
    print_reward(env, print_freq=1000, metric_name="midpoint_to_target_distance", metric_value=distance, reward_name="midpoint_to_target_distance_reward", reward_value=dist_reward)
    return dist_reward

def finger_closure_reward(env: ManagerBasedRLEnv, 
                          left_finger_cfg: SceneEntityCfg,
                          right_finger_cfg: SceneEntityCfg,
                          target_width: float,
                          activation_dist: float,
                          std_gap: float) -> torch.Tensor:
    left_finger_pos_w, right_finger_pos_w, midpoint_pos_w = get_finger_features(env, left_finger_cfg, right_finger_cfg)
    
    gap = torch.norm(left_finger_pos_w - right_finger_pos_w, dim=1)
    target_gap = target_width * 0.99
    gap_error = torch.abs(gap - target_gap)
    
    cube_w = env.scene["cube"].data.root_pos_w[:, :3]
    dist = torch.norm(midpoint_pos_w - cube_w, dim=1)
    
    closure_reward = torch.clamp(1.0 - gap_error / std_gap, min=0.0, max=1.0)
    
    reward = torch.where(
        dist <= activation_dist,
        closure_reward,
        torch.zeros_like(closure_reward)
    )
    print_reward(env, print_freq=1000, metric_name="dist", metric_value=dist, reward_name="closure_reward", reward_value=closure_reward)
    print_reward(env, print_freq=1000, metric_name="gap", metric_value=gap, reward_name="closure_reward", reward_value=closure_reward)
    print_reward(env, print_freq=1000, metric_name="gap_error", metric_value=gap_error, reward_name="reward", reward_value=reward)
    
    return reward

def finger_height_alignment_reward(
    env: ManagerBasedRLEnv,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    target_asset_cfg: SceneEntityCfg,
    std_height: float = 0.02,
) -> torch.Tensor:
    """Reward when fingers are at similar height as cube center (side grasp).
    
    This is the KEY reward to distinguish side grasp vs top press!
    """
    
    target_asset = env.scene[target_asset_cfg.name]
    
    cube_pos = target_asset.data.root_pos_w[:, :3]
    _, _, midpoint_pos_w = get_finger_features(env, left_finger_cfg, right_finger_cfg)
    midpoint_z = midpoint_pos_w[:,2]
    cube_z = cube_pos[:, 2]
    
    height_diff = torch.abs(midpoint_z - cube_z)
    
    reward = torch.exp(-height_diff / std_height)
    
    if env.unwrapped.common_step_counter % 10000 == 0:
        print(f"[DEBUG] Height Alignment - "
              f"mean diff: {height_diff.mean().item():.4f}m, "
              f"reward: {reward.mean().item():.4f}, "
              f">0.04m (top press): {(height_diff > 0.04).sum().item()}/{env.num_envs}")
    
    return reward


# def native_finger_grasp_reward(env: ManagerBasedRLEnv, std_dist, std_grasp, left_finger_cfg, right_finger_cfg, target_asset_cfg) -> torch.Tensor:
#     target_asset: RigidObject = env.scene[target_asset_cfg.name]
#     target_asset_pos_w = target_asset.data.root_pos_w[:, :3]
#     left_finger_pos_w, right_finger_pos_w, midpoint_pos_w = get_finger_features(env, left_finger_cfg, right_finger_cfg)
    
#     distance = get_target_distance(env, target_asset_cfg, midpoint_pos_w)
#     dist_factor = torch.exp(-distance / std_dist)
    
#     left_to_target = left_finger_pos_w - target_asset_pos_w
#     right_to_target = right_finger_pos_w - target_asset_pos_w
#     left_to_target_len = torch.norm(left_to_target, dim=1)
#     right_to_target_len = torch.norm(right_to_target, dim=1)
#     eps = 1e-6
#     left_vec_norm = left_to_target / (left_to_target_len.unsqueeze(-1) + eps)
#     right_vec_norm = right_to_target / (right_to_target_len.unsqueeze(-1) + eps)

#     length_diff = torch.abs(left_to_target_len - right_to_target_len)
#     length_reward = torch.exp(-length_diff / std_grasp)
    
#     dot_product = torch.sum(left_vec_norm * right_vec_norm, dim=1)
#     opposite_error = torch.abs(dot_product + 1.0)
#     opposite_reward = torch.exp(-opposite_error*3.0)
    
#     grasp_quality = length_reward * opposite_reward * dist_factor
#     # grasp_quality = opposite_reward * dist_factor
#     if env.unwrapped.common_step_counter % 10000 == 0:
#             print("[native_finger_grasp_reward]:")
#             print_reward(env, print_freq=1000, metric_name="length_diff", metric_value=length_diff, reward_name="length_reward", reward_value=length_reward)
#             print_reward(env, print_freq=1000, metric_name="opposite_error", metric_value=opposite_error, reward_name="opposite_reward", reward_value=opposite_reward)
#             print_reward(env, print_freq=1000, metric_name="distance", metric_value=distance, reward_name="dist_factor", reward_value=dist_factor)
#     return grasp_quality


def finger_symmetry_reward(env, target_asset_cfg, left_finger_cfg, right_finger_cfg, std_grasp):
    target_pos = env.scene[target_asset_cfg.name].data.root_pos_w[:, :3]
    left_finger_pos_w, right_finger_pos_w, midpoint_pos_w = get_finger_features(env, left_finger_cfg, right_finger_cfg)
    l_dist = torch.norm(left_finger_pos_w - target_pos, dim=1)
    r_dist = torch.norm(right_finger_pos_w - target_pos, dim=1)
    diff = torch.abs(l_dist - r_dist)
    return torch.exp(-diff / std_grasp)

def finger_opposition_reward(env, target_asset_cfg, left_finger_cfg, right_finger_cfg):
    target_pos = env.scene[target_asset_cfg.name].data.root_pos_w[:, :3]
    left_finger_pos_w, right_finger_pos_w, midpoint_pos_w = get_finger_features(env, left_finger_cfg, right_finger_cfg)
    v_left = left_finger_pos_w - target_pos
    v_right = right_finger_pos_w - target_pos
    v_left_n = v_left / (torch.norm(v_left, dim=1, keepdim=True) + 1e-6)
    v_right_n = v_right / (torch.norm(v_right, dim=1, keepdim=True) + 1e-6)
    dot = torch.sum(v_left_n * v_right_n, dim=1)
    return torch.exp(-torch.abs(dot + 1.0) * 3.0)

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

def position_command_error_tanh(
        env: ManagerBasedRLEnv, std_dist: float, command_name: str, asset_cfg: SceneEntityCfg, left_finger_cfg, right_finger_cfg
    ) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w = env.scene.env_origins + des_pos_b
    asset_curr_pos_w = asset.data.root_pos_w[:, :3]
    distance = torch.norm(asset_curr_pos_w - des_pos_w, dim=1)
    dist_reward = torch.exp(-distance / std_dist)
    debug_cube_move_state(env, distance, dist_reward, print_freq_stats=1000)
    print_reward(env, print_freq=1000, metric_name="cube_move_distance", metric_value=distance, reward_name="cube_move_position_tracking_reward", reward_value=dist_reward)
    
    # Only enable the reward when the grippe is grasping the asset
    _, _, midpoint_pos_w = get_finger_features(env, left_finger_cfg, right_finger_cfg)
    grasp_dist = get_target_distance(env, asset_cfg, midpoint_pos_w)
    grasp_factor = torch.exp(-grasp_dist / std_dist)
    
    reward = dist_reward * grasp_factor
    if env.unwrapped.common_step_counter % 1000 == 0:
        print(f"[DEBUG] Grasp Dist Mean: {grasp_dist.mean().item():.4f}, Mask Mean: {grasp_factor.mean().item():.2f}")
    return reward

def asset_vel_to_command(
        env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, cube_length: float
    ) -> torch.Tensor:
    (alignment_score, _, _, _, _, to_target_dist) = compute_cube_velocity_alignment(env, asset_cfg, command_name)
    result = torch.where(
        to_target_dist < cube_length / 2, 
        torch.ones_like(alignment_score), 
        alignment_score                   
    ).squeeze()
    
    return result

def debug_gripper_y_axis(env, ee_link_cfg, target_asset_cfg, print_freq=1000):
    if env.unwrapped.common_step_counter % print_freq != 0:
        return
    
    dot_product = wrist_outside_normal_to_target(env, ee_link_cfg, target_asset_cfg)
    angle_rad = torch.acos(dot_product)
    angle_deg = angle_rad * 180.0 / math.pi
    
    print(f"[DEBUG] Step {env.unwrapped.common_step_counter}:")
    print(f"  Y-Axis to Approach Angle:")
    print(f"    dot_product - mean: {dot_product.mean().item():.3f}, "
          f"std: {dot_product.std().item():.3f}")
    print(f"    angle (deg) - mean: {angle_deg.mean().item():.1f}°, "
          f"min: {angle_deg.min().item():.1f}°, max: {angle_deg.max().item():.1f}°")
    print(f"    good alignment (<30°): {(angle_deg < 30).sum().item()}/{env.num_envs}")
    print(f"    bad alignment (>60°): {(angle_deg > 60).sum().item()}/{env.num_envs}")

def wrist_outside_normal_to_target_reward(
    env: ManagerBasedRLEnv,
    ee_link_cfg: SceneEntityCfg,
    target_asset_cfg: SceneEntityCfg,
    std_angle: float = 0.5,
) -> torch.Tensor:

    dot_product = wrist_outside_normal_to_target(env, ee_link_cfg, target_asset_cfg)
    reward = torch.exp(-(1 - dot_product) ** 2 / std_angle)
    
    if env.unwrapped.common_step_counter % 10000 == 0:
        print(f"[DEBUG] Y-Axis Approach - "
              f"dot mean: {dot_product.mean().item():.3f}, "
              f"reward mean: {reward.mean().item():.3f}, "
              f"good (>0.8): {(reward > 0.8).sum().item()}/{env.num_envs}")
        debug_gripper_y_axis(env, ee_link_cfg, target_asset_cfg)
    
    return reward


def finger_line_horizontal_reward(
    env: "ManagerBasedRLEnv",
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    std_angle: float = 0.3,
) -> torch.Tensor:
    info = get_finger_line_horizontal_info(env, left_finger_cfg, right_finger_cfg)
    reward = torch.exp(-(info["angle_rad"] ** 2) / std_angle)
    
    if env.unwrapped.common_step_counter % 10000 == 0:
        print(f"[DEBUG] Finger Line Horizontal - "
              f"angle mean: {info['angle_deg'].mean().item():.1f}°, "
              f"reward mean: {reward.mean().item():.3f}, "
              f"good (<15°): {(info['angle_deg'] < 15).sum().item()}/{env.num_envs}")
    return reward