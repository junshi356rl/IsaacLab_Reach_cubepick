import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.assets import RigidObject
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul
from .helper import compute_gripper_midpoint_dot, compute_cube_velocity_alignment, get_to_target, wrist_outside_normal_to_target, get_finger_line_horizontal_info, get_offset_body_pos_w, PAD_OFFSET
from .....helpers.robotiq_fingertip_center_helper import get_left_right_fingertip_midpoint_pos_w

def position_target_asset_delta_vector(env, asset_cfg, target_asset_cfg):
    """Returns the position error vector from asset to target."""
    asset: RigidObject = env.scene[asset_cfg.name]
    target_asset: RigidObject = env.scene[target_asset_cfg.name]

    # Get current positions in world frame
    if hasattr(asset_cfg, 'body_ids') and asset_cfg.body_ids is not None:
        curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]
    else:
        curr_pos_w = asset.data.body_state_w[:, 0, :3]  # root

    target_pos_w = target_asset.data.body_state_w[:, 0, :3]

    return target_pos_w - curr_pos_w  # shape: (N, 3)

def position_target_asset_delta_vector_norm(env, asset_cfg, target_asset_cfg):
    delta_pos = position_target_asset_delta_vector(env, asset_cfg, target_asset_cfg)
    return torch.norm(delta_pos, dim=1, keepdim=True)

def orientation_target_link(env, asset_cfg):
    asset: RigidObject = env.scene[asset_cfg.name]
    if hasattr(asset_cfg, 'body_ids') and asset_cfg.body_ids is not None:
        quat = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]
    else:
        raise ValueError("asset_cfg must have body_ids defined for orientation_target_link")
    return quat

# return the forces on a contact sensor
def contact_sensor_forces(env, sensor_cfg):
    sensor = env.scene[sensor_cfg.name]
    # flatten the forces to (N, contacts*3)
    flattened = sensor.data.force_matrix_w.view(env.num_envs, -1)
    return flattened  # shape: (N, 3)

def gripper_to_target_angle_obs(
        env,
        left_finger_cfg: "SceneEntityCfg" = None,
        right_finger_cfg: "SceneEntityCfg" = None,
        target_asset_cfg: "SceneEntityCfg" = None,
    ):
    
    dot_product = compute_gripper_midpoint_dot(env, left_finger_cfg, right_finger_cfg, target_asset_cfg)
    # Reward: high when |dot| ≈ 0
    # reward = 1.0 - torch.tanh((dot_product ** 2) / std)
    angle = torch.acos(torch.abs(dot_product))
    return angle.unsqueeze(-1)

def fingertip_midpoint_to_target_vector(env, target_asset_cfg):
    gripper_fingertip_midpoint = get_left_right_fingertip_midpoint_pos_w(env)
    target_asset: RigidObject = env.scene[target_asset_cfg.name]
    target_pos_w = target_asset.data.body_state_w[:, 0, :3]
    return target_pos_w - gripper_fingertip_midpoint

def inner_finger_gap_minus_cube_length_native(env, left_finger_cfg, right_finger_cfg, cube_length):
    left_finger_asset = env.scene[left_finger_cfg.name]
    right_finger_asset = env.scene[right_finger_cfg.name]
    left_tip_pos_w = get_offset_body_pos_w(left_finger_asset, left_finger_cfg.body_names[0], PAD_OFFSET)
    right_tip_pos_w = get_offset_body_pos_w(right_finger_asset, right_finger_cfg.body_names[0], PAD_OFFSET)
    return torch.norm(left_tip_pos_w - right_tip_pos_w, dim=1, keepdim=True) - cube_length

def each_finger_to_target_native(env, left_finger_cfg, right_finger_cfg, target_asset_cfg):
    left_finger_asset = env.scene[left_finger_cfg.name]
    right_finger_asset = env.scene[right_finger_cfg.name]
    left_finger_pos_w = get_offset_body_pos_w(left_finger_asset, left_finger_cfg.body_names[0], PAD_OFFSET)
    right_finger_pos_w = get_offset_body_pos_w(right_finger_asset, right_finger_cfg.body_names[0], PAD_OFFSET)
    target_asset: RigidObject = env.scene[target_asset_cfg.name]
    target_asset_pos_w = target_asset.data.body_state_w[:, 0, :3]
    left_finger_target_vector = target_asset_pos_w - left_finger_pos_w
    rightfinger_target_vector = target_asset_pos_w - right_finger_pos_w
    return torch.concat((left_finger_target_vector, rightfinger_target_vector), dim=1)

def finger_midpoint_to_target_native(env, left_finger_cfg, right_finger_cfg, target_asset_cfg):
    (mid_point, to_target) = get_to_target(env, left_finger_cfg, right_finger_cfg, target_asset_cfg)
    return torch.concat((mid_point, to_target), dim=1)

def finger_quat_native(env, left_finger_cfg, right_finger_cfg):
    left_finger_asset = env.scene[left_finger_cfg.name]
    right_finger_asset = env.scene[right_finger_cfg.name]
    left_finger_quat_w = left_finger_asset.data.body_quat_w[:, left_finger_cfg.body_ids[0], :]
    right_finger_quat_w = right_finger_asset.data.body_quat_w[:, right_finger_cfg.body_ids[0], :]
    return torch.concat((left_finger_quat_w, right_finger_quat_w), dim=1)

def asset_to_command_vector(env, target_asset_cfg, command_name):
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w = env.scene.env_origins + des_pos_b
    cube_pos_w = env.scene[target_asset_cfg.name].data.root_pos_w
    return des_pos_w - cube_pos_w

def get_asset_vel(env, asset_cfg):
    asset = env.scene[asset_cfg.name]
    return asset.data.root_vel_w

def get_body_vel(env, body_cfg):
    body_idx = env.scene[body_cfg.name].find_bodies(body_cfg.body_names[0])[0][0]
    return env.scene[body_cfg.name].data.body_vel_w[:, body_idx, :3]

def get_env_origin(env):
    res = env.scene.env_origins
    return res

def get_cube_velocity_alignment(env, asset_cfg, command_name):
    (alignment_score, _, _, _, _, _) = compute_cube_velocity_alignment(env, asset_cfg, command_name)
    return alignment_score


def wrist_outside_normal_to_target_rad(
    env: "ManagerBasedRLEnv",
    ee_link_cfg: SceneEntityCfg,
    target_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:

    dot_product = wrist_outside_normal_to_target(env, ee_link_cfg, target_asset_cfg)
    
    angle_rad = torch.acos(dot_product)  # [0, π/2]
    return angle_rad.unsqueeze(-1)

def finger_line_horizontal_obs(
    env: "ManagerBasedRLEnv",
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
) -> torch.Tensor:
    info = get_finger_line_horizontal_info(env, left_finger_cfg, right_finger_cfg)
    return info["angle_rad"].unsqueeze(-1)