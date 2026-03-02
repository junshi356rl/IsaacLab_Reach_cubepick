import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.assets import RigidObject
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul
from .helper import compute_gripper_midpoint_dot
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

def inner_finger_gap_native(env, left_finger_cfg, right_finger_cfg):
    left_finger_asset = env.scene[left_finger_cfg.name]
    right_finger_asset = env.scene[right_finger_cfg.name]
    left_tip_pos_w = left_finger_asset.data.body_pos_w[:, left_finger_cfg.body_ids[0], :3]
    right_tip_pos_w = right_finger_asset.data.body_pos_w[:, right_finger_cfg.body_ids[0], :3]
    return torch.norm(left_tip_pos_w - right_tip_pos_w, dim=1, keepdim=True)

def inner_finger_midpoint_to_target_native(env, left_finger_cfg, right_finger_cfg, target_asset_cfg):
    left_finger_asset = env.scene[left_finger_cfg.name]
    right_finger_asset = env.scene[right_finger_cfg.name]
    left_finger_pos_w = left_finger_asset.data.body_pos_w[:, left_finger_cfg.body_ids[0], :3]
    right_finger_pos_w = right_finger_asset.data.body_pos_w[:, right_finger_cfg.body_ids[0], :3]
    target_asset: RigidObject = env.scene[target_asset_cfg.name]
    target_asset_pos_w = target_asset.data.body_state_w[:, 0, :3]
    midpoint_pos_w = (left_finger_pos_w + right_finger_pos_w) / 2.0
    return target_asset_pos_w - midpoint_pos_w