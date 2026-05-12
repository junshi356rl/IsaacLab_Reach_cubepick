from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from isaaclab.utils.math import normalize, quat_apply
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

PAD_OFFSET = (0.0, 0.045755401253700256, -0.027220344170928)

def get_finger_axis(
    env: "ManagerBasedRLEnv",
    left_finger_cfg,
    right_finger_cfg,
) -> torch.Tensor:
    """Compute the gripper opening axis (projected to XY).

    Returns a normalized axis per environment (shape: (N, 3)).
    """
    left_finger_asset = env.scene[left_finger_cfg.name]
    right_finger_asset = env.scene[right_finger_cfg.name]

    left_body_id = left_finger_asset.find_bodies(left_finger_cfg.body_names[0])[0]
    right_body_id = right_finger_asset.find_bodies(right_finger_cfg.body_names[0])[0]

    left_pad_pos_w = get_offset_body_pos_w(left_finger_asset, left_finger_cfg.body_names[0], PAD_OFFSET)
    right_pad_pos_w = get_offset_body_pos_w(right_finger_asset, right_finger_cfg.body_names[0], PAD_OFFSET)

    finger_axis = (right_pad_pos_w - left_pad_pos_w).squeeze()

    return left_pad_pos_w, finger_axis

def get_to_target(env: "ManagerBasedRLEnv",
    left_finger_cfg,
    right_finger_cfg,
    target_asset_cfg,
) -> torch.Tensor:
    """Compute the direction from the fingers' midpoint to the target (projected to XY).

    Returns a normalized axis per environment (shape: (N, 3)).
    """
    left_finger_asset = env.scene[left_finger_cfg.name]
    right_finger_asset = env.scene[right_finger_cfg.name]

    left_pad_pos_w = get_offset_body_pos_w(left_finger_asset, left_finger_cfg.body_names[0], PAD_OFFSET)
    right_pad_pos_w = get_offset_body_pos_w(right_finger_asset, right_finger_cfg.body_names[0], PAD_OFFSET)

    mid_point = (left_pad_pos_w + right_pad_pos_w) / 2
    target_asset = env.scene[target_asset_cfg.name]
    target_pos = target_asset.data.body_state_w[:, 0, :3]
    to_target = target_pos - mid_point
    return mid_point, to_target

def compute_gripper_midpoint_dot(
    env: "ManagerBasedRLEnv",
    left_finger_cfg,
    right_finger_cfg,
    target_asset_cfg,
) -> torch.Tensor:
    """Compute the dot product between the gripper opening axis (projected to XY)
    and the direction from the fingers' midpoint to the target (projected to XY).

    Returns a clamped dot product per environment (shape: (N,)).
    """
    _, finger_axis = get_finger_axis(env, left_finger_cfg, right_finger_cfg)
    finger_axis_xy = finger_axis.clone()
    finger_axis_xy[..., 2] = 0.0
    finger_axis_xy = normalize(finger_axis_xy)

    _, to_target = get_to_target(env, left_finger_cfg, right_finger_cfg, target_asset_cfg)
    to_target_xy = to_target.clone()
    to_target_xy[..., 2] = 0.0
    to_target_xy = normalize(to_target_xy)

    dot_product = torch.sum(finger_axis_xy * to_target_xy, dim=1)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    return dot_product

def get_body_position(env, asset_name, body_name):
    asset = env.scene[asset_name]
    body_id = asset.find_bodies(body_name)[0]
    pos = asset.data.body_state_w[:, body_id, :3].squeeze()
    return pos

def get_body_quat(env, asset_name, body_name):
    asset = env.scene[asset_name]
    body_id = asset.find_bodies(body_name)[0]
    quat = asset.data.body_state_w[:, body_id, 3:7].squeeze()
    return quat

def compute_cube_velocity_alignment(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    eps: float = 1e-6
) -> dict:
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    des_pos_b = command[:, :3]
    des_pos_w = env.scene.env_origins + des_pos_b
    
    curr_pos_w = asset.data.root_pos_w[:, :3]
    
    to_target = des_pos_w - curr_pos_w
    to_target_dist = torch.norm(to_target, dim=1, keepdim=True) + eps
    to_target_dir = to_target / to_target_dist
    
    cube_vel = asset.data.root_vel_w[:, :3]  # (N, 3)
    cube_speed = torch.norm(cube_vel, dim=1, keepdim=True)  # (N, 1)
    
    vel_dir = cube_vel / (cube_speed + eps)  # (N, 3)
    
    alignment = torch.sum(to_target_dir * vel_dir, dim=1, keepdim=True)
    
    return (alignment, to_target_dir, vel_dir, cube_speed, cube_vel, to_target_dist)

def wrist_outside_normal_to_target(
    env: ManagerBasedRLEnv,
    ee_link_cfg: SceneEntityCfg,
    target_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:

    ee_asset = env.scene[ee_link_cfg.name]
    ee_body_id = ee_asset.find_bodies(ee_link_cfg.body_names[0])[0]
    ee_quat_w = ee_asset.data.body_quat_w[:, ee_body_id, :]  # (N, 4), [w, x, y, z]
    
    ee_pos_w = ee_asset.data.body_pos_w[:, ee_body_id, :3].squeeze(1)  # (N, 3)
    
    target_asset = env.scene[target_asset_cfg.name]
    target_pos_w = target_asset.data.root_pos_w[:, :3]  # (N, 3)
    
    approach_vec = target_pos_w - ee_pos_w  # (N, 3)
    approach_dist = torch.norm(approach_vec, dim=1, keepdim=True) + 1e-6
    approach_vec_norm = approach_vec / approach_dist  # (N, 3)
    
    
    ee_local_z_axis = torch.tensor([0.0, 0.0, 1.0], device=ee_quat_w.device).unsqueeze(0).expand(ee_quat_w.shape[0], -1)
    ee_world_z_axis = quat_apply(ee_quat_w, ee_local_z_axis)  # (N, 3)
    dot_product = torch.sum(ee_world_z_axis * approach_vec_norm, dim=1)  # (N,)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    
    return dot_product


def get_offset_body_pos_w(
    asset,
    body_name: str,
    local_offset: tuple[float, float, float] | torch.Tensor,
) -> torch.Tensor:
    body_idx = asset.find_bodies(body_name)[0]
    if isinstance(body_idx, list):
        body_idx = body_idx[0]

    pos_w = asset.data.body_pos_w[:, body_idx, :3].squeeze(1)
    quat_w = asset.data.body_quat_w[:, body_idx, :].squeeze(1)

    offset_tensor = torch.as_tensor(local_offset, device=pos_w.device)
    if offset_tensor.ndim == 1:
        offset_tensor = offset_tensor.unsqueeze(0).expand(pos_w.shape[0], -1)

    return pos_w + quat_apply(quat_w, offset_tensor)

def get_finger_line_horizontal_info(
    env: "ManagerBasedRLEnv",
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
) -> dict:
    left_finger_asset = env.scene[left_finger_cfg.name]
    right_finger_asset = env.scene[right_finger_cfg.name]
    
    left_body_id = left_finger_asset.find_bodies(left_finger_cfg.body_names[0])[0]
    right_body_id = right_finger_asset.find_bodies(right_finger_cfg.body_names[0])[0]

    left_pad_pos_w = get_offset_body_pos_w(left_finger_asset, left_finger_cfg.body_names[0], PAD_OFFSET)
    right_pad_pos_w = get_offset_body_pos_w(right_finger_asset, right_finger_cfg.body_names[0], PAD_OFFSET)

    line_vector = right_pad_pos_w - left_pad_pos_w
    line_vector_norm = normalize(line_vector)
    
    z_component = torch.abs(line_vector_norm[:, 2])
    z_component = torch.clamp(z_component, 0.0, 1.0)
    
    angle_rad = torch.asin(z_component)
    angle_deg = angle_rad * 180.0 / torch.pi
    
    xy_projection = torch.sqrt(line_vector_norm[:, 0]**2 + line_vector_norm[:, 1]**2)
    
    return {
        "line_vector": line_vector,
        "line_vector_norm": line_vector_norm,
        "z_component": z_component,
        "xy_projection": xy_projection,
        "angle_deg": angle_deg,
        "angle_rad": angle_rad,
    }


def cube_ee_relative_vel(env, ee_link_name: str) -> torch.Tensor:
    """Compute relative velocity between cube and end-effector (cube_vel - ee_vel)."""
    cube_vel = env.scene["cube"].data.root_vel_w[:, :3]
    ee_body_id = env.scene["robot"].find_bodies(ee_link_name)[0]
    ee_vel = env.scene["robot"].data.body_vel_w[:, ee_body_id, :3].squeeze(1)
    return cube_vel - ee_vel