from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from isaaclab.utils.math import normalize

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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

    left_pos = left_finger_asset.data.body_state_w[:, left_body_id, :3].squeeze()
    right_pos = right_finger_asset.data.body_state_w[:, right_body_id, :3].squeeze()

    finger_axis = (right_pos - left_pos).squeeze()

    return left_pos, finger_axis

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

    left_body_id = left_finger_asset.find_bodies(left_finger_cfg.body_names[0])[0]
    right_body_id = right_finger_asset.find_bodies(right_finger_cfg.body_names[0])[0]

    left_pos = left_finger_asset.data.body_state_w[:, left_body_id, :3].squeeze()
    right_pos = right_finger_asset.data.body_state_w[:, right_body_id, :3].squeeze()

    mid_point = (left_pos + right_pos) / 2
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