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
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    command = env.command_manager.get_command(command_name)

    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]

    return quat_error_magnitude(curr_quat_w, des_quat_w)


def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    if hasattr(asset_cfg, 'body_ids') and asset_cfg.body_ids is not None and asset_cfg.body_ids.start is not None:
        curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3] # type: ignore
    else:
        curr_pos_w = asset.data.body_state_w[:, [0], :3].squeeze() # type: ignore
    pos_error = curr_pos_w - des_pos_w
    return torch.norm(pos_error, dim=1)

def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
    ) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3] # type: ignore
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