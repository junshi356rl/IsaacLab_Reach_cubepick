import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.assets import RigidObject
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

def position_target_asset_error_vector(env, asset_cfg, target_asset_cfg):
    """Returns the position error vector from asset to target."""
    asset: RigidObject = env.scene[asset_cfg.name]
    target_asset: RigidObject = env.scene[target_asset_cfg.name]

    # Get current positions in world frame
    if hasattr(asset_cfg, 'body_ids') and asset_cfg.body_ids is not None:
        curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]
    else:
        curr_pos_w = asset.data.body_state_w[:, 0, :3]  # root

    target_pos_w = target_asset.data.body_state_w[:, 0, :3]

    return curr_pos_w - target_pos_w  # shape: (N, 3)
