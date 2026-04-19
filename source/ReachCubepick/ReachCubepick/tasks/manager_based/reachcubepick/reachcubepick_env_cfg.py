# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import random
import isaaclab.sim as sim_utils
import isaaclab.assets
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    ActionTermCfg as ActionTerm,
    CurriculumTermCfg as CurrTerm,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.sim.spawners.shapes import CuboidCfg
from isaaclab.sim.spawners.materials import PhysicsMaterialCfg,RigidBodyMaterialCfg
import torch

from . import mdp
from ....helpers.robotiq_fingertip_center_helper import get_left_right_fingertip_gap

from .ur_gripper import UR_GRIPPER_CFG, UR_PATH, BASE_LINK_NAME, EE_LINK_NAME, ROBOT_PRIM_NAME, GRIPPER_PRIM_NAME
import isaaclab.sim.schemas
import carb.settings
from pxr import Usd

def get_random_translation():
    x = random.uniform(0.3, 0.6)
    y = random.uniform(0.1, 0.2)
    z = CUBE_LENGTH/2 + 0.001  # Slightly above the ground to avoid initial penetration
    if random.random() < 0.5:
        y = -y

    return (x, y, z)

def read_meters_per_unit_from_usd(file_path: str) -> float:
    stage = Usd.Stage.Open(file_path)
    scale = stage.GetMetadata('metersPerUnit')
    return scale if scale is not None else 1.0

##
# Scene definition
##

ENV_SPACING = 2.5
CUBE_LENGTH = 0.08
DIST_TOLERANCE = CUBE_LENGTH/5
GRASP_TOLERANCE = CUBE_LENGTH/10
CUBE_MASS = 0.5
unit_scale = read_meters_per_unit_from_usd(UR_PATH)
INNER_FINGER_SIZE = [unit_scale*0.0655, 0, 0] # https://blog.robotiq.com/hubfs/support-files/2F-85_2F-140_UR_PDF_20240402.pdf
MIN_FINGER_GAP = 0.01
MAX_FINGER_GAP = 0.14
EPISODE_LENGTH_S = 6.0
STD_DIST = 0.15
STD_GRASP = CUBE_LENGTH/4
STD_DIST_MOVE = 0.04

@configclass
class ReachcubepickSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot = UR_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=CuboidCfg(
            size=(CUBE_LENGTH, CUBE_LENGTH, CUBE_LENGTH),
            mass_props=sim_utils.schemas.MassPropertiesCfg(mass=CUBE_MASS),
            rigid_props=sim_utils.schemas.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material = RigidBodyMaterialCfg(
                static_friction = 0.8,
                dynamic_friction = 0.7,
                restitution = 0.1),
        ),
        init_state = RigidObjectCfg.InitialStateCfg(pos=get_random_translation()),
    )
    left_finger_contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}"+f"/Robot/{ROBOT_PRIM_NAME}/{GRIPPER_PRIM_NAME}/left_inner_finger",
        update_period=0.01,
        history_length=3,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"]
    )
    right_finger_contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}"+f"/Robot/{ROBOT_PRIM_NAME}/{GRIPPER_PRIM_NAME}/right_inner_finger",
        update_period=0.01,
        history_length=3,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"]
    )

##
# MDP settings
##

@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        # Robot observations
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # joint_effort = ObsTerm(func=mdp.joint_effort)
        # Action and Command
        actions = ObsTerm(func=mdp.last_action)
        
        # Cube observations
        cube_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("cube")}) # TODO: should use pos relative to robot base
        cube_quat = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("cube")})
        cube_vel = ObsTerm(func=mdp.get_asset_vel, params={"asset_cfg": SceneEntityCfg("cube")})
        
        ee_pos = ObsTerm(func=mdp.body_pose_w, params={"asset_cfg": SceneEntityCfg("robot", body_names=[EE_LINK_NAME])})
        # ee_vel = ObsTerm(func=mdp.get_body_vel, params={"body_cfg": SceneEntityCfg("robot", body_names=[EE_LINK_NAME])})
        gripper_y_axis_approach = ObsTerm(
                func=mdp.wrist_outside_normal_to_target_rad,
                params={
                    "ee_link_cfg": SceneEntityCfg("robot", body_names=[EE_LINK_NAME]),
                    "target_asset_cfg": SceneEntityCfg("cube"),
                }
            )
        finger_line_horizontal = ObsTerm(
            func=mdp.finger_line_horizontal_obs,
            params={
                "left_finger_cfg": SceneEntityCfg("robot", body_names=["left_inner_finger"]),
                "right_finger_cfg": SceneEntityCfg("robot", body_names=["right_inner_finger"]),
            }
        )
        env_origin = ObsTerm(func=mdp.get_env_origin)
        finger_gap_native = ObsTerm(func=mdp.inner_finger_gap_minus_cube_length_native, params={
            "left_finger_cfg": SceneEntityCfg("robot", body_names=["left_inner_finger"]),
            "right_finger_cfg": SceneEntityCfg("robot", body_names=["right_inner_finger"]),
            "cube_length": CUBE_LENGTH
        })
        # cube_fingertip_mid_diff = ObsTerm(func=mdp.fingertip_midpoint_to_target_vector, params={
        #     "target_asset_cfg": SceneEntityCfg("cube")})
        # finger_to_cube_vel_native = ObsTerm(func=mdp.inner_finger_midpoint_vel_to_target_native, params={
        #     "left_finger_cfg": SceneEntityCfg("robot", body_names=["left_inner_finger"]),
        #     "right_finger_cfg": SceneEntityCfg("robot", body_names=["right_inner_finger"]),
        #     "target_asset_cfg": SceneEntityCfg("cube")
        # })
        each_finger_to_target_native = ObsTerm(func=mdp.each_finger_to_target_native, params={
            "left_finger_cfg": SceneEntityCfg("robot", body_names=["left_inner_finger"]),
            "right_finger_cfg": SceneEntityCfg("robot", body_names=["right_inner_finger"]),
            "target_asset_cfg": SceneEntityCfg("cube")
        })
        finger_midpoint_to_target_native = ObsTerm(func=mdp.finger_midpoint_to_target_native, params={
            "left_finger_cfg": SceneEntityCfg("robot", body_names=["left_inner_finger"]),
            "right_finger_cfg": SceneEntityCfg("robot", body_names=["right_inner_finger"]),
            "target_asset_cfg": SceneEntityCfg("cube")
        })
        finger_quat_native = ObsTerm(func=mdp.finger_quat_native, params={
            "left_finger_cfg": SceneEntityCfg("robot", body_names=["left_inner_finger"]),
            "right_finger_cfg": SceneEntityCfg("robot", body_names=["right_inner_finger"]),
        })
        left_finger_sensor_forces = ObsTerm(
            func=mdp.contact_sensor_forces,
            params={"sensor_cfg": SceneEntityCfg(name="left_finger_contact_sensor")},
        )
        right_finger_sensor_forces = ObsTerm(
            func=mdp.contact_sensor_forces,
            params={"sensor_cfg": SceneEntityCfg(name="right_finger_contact_sensor")},
        )
        # finger_cube_rel_vel = ObsTerm(
        #     func=mdp.inner_finger_midpoint_vel_to_target_native,
        #     params={
        #         "left_finger_cfg": SceneEntityCfg("robot", body_names=["left_inner_finger"]),
        #         "right_finger_cfg": SceneEntityCfg("robot", body_names=["right_inner_finger"]),
        #         "target_asset_cfg": SceneEntityCfg("cube")
        #     }
        # )
        move_target_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "move_target"})
        cube_to_command = ObsTerm(
            func=mdp.asset_to_command_vector,
            params={
                "target_asset_cfg":SceneEntityCfg("cube"),
                "command_name":"move_target"
            }
        )
        # cube_velocity_alignment = ObsTerm(
        #     func=mdp.get_cube_velocity_alignment,
        #     params={
        #         "asset_cfg":SceneEntityCfg("cube"),
        #         "command_name":"move_target"
        #     }
        # )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    arm_action: ActionTerm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
        use_default_offset=True,
        debug_vis=True
    )
    gripper_action: ActionTerm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        use_default_offset=True,
        debug_vis=True
    )
    # gripper_action: ActionTerm = mdp.JointEffortActionCfg(
    #     asset_name="robot",
    #     joint_names=["finger_joint"],
    #     scale=50.0,
    #     debug_vis=True
    # )

@configclass
class CommandsCfg:
    move_target = mdp.UniformPoseCommandCfg(
        asset_name="robot", # target is based on the robot root
        body_name=BASE_LINK_NAME,
        resampling_time_range=(EPISODE_LENGTH_S, EPISODE_LENGTH_S),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.6, 0.8),
            pos_y=(-0.5, -0.3),
            pos_z=(0.1, 0.3),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )

@configclass
class RewardsCfg:
    # gripper_cube_dist_reward = RewTerm(
    #     func=mdp.gripper_target_dist_reward,
    #     weight=3.0,
    #     params={
    #         'std_dist': STD_DIST,
    #         'target_asset_cfg': SceneEntityCfg("cube"),
    #     }
    # )
    gripper_cube_dist = RewTerm(
        func=mdp.native_finger_midpoint_to_target_distance_reward,
        weight=5.0,
        params={
            'std_dist': STD_DIST,
            'left_finger_cfg': SceneEntityCfg("robot", body_names=["left_inner_finger"]),
            'right_finger_cfg': SceneEntityCfg("robot", body_names=["right_inner_finger"]),
            'target_asset_cfg': SceneEntityCfg("cube"),
        }
    )

    # finger_cube_orien_rel = RewTerm(
    #     func=mdp.native_finger_grasp_reward,
    #     weight=5.0,
    #     params={
    #         'std_dist': STD_DIST,
    #         'std_grasp': STD_GRASP,
    #         'left_finger_cfg': SceneEntityCfg("robot", body_names=["left_inner_finger"]),
    #         'right_finger_cfg': SceneEntityCfg("robot", body_names=["right_inner_finger"]),
    #         'target_asset_cfg': SceneEntityCfg("cube"),
    #     }
    # )

    # finger_gap = RewTerm(
    #     func=mdp.native_finger_gap_reward,
    #     weight=5.0,
    #     params={
    #         'cube_length': CUBE_LEGNTH,
    #         'std_dist': STD_DIST,
    #         'left_finger_cfg': SceneEntityCfg("robot", body_names=["left_inner_finger"]),
    #         'right_finger_cfg': SceneEntityCfg("robot", body_names=["right_inner_finger"]),
    #         'target_asset_cfg': SceneEntityCfg("cube"),
    #     }
    # )

    finger_symmetry = RewTerm(
        func=mdp.finger_symmetry_reward,
        weight=3.0,
        params={
            'std_grasp': STD_GRASP,
            'target_asset_cfg': SceneEntityCfg("cube"),
            'left_finger_cfg': SceneEntityCfg("robot", body_names=["left_inner_finger"]),
            'right_finger_cfg': SceneEntityCfg("robot", body_names=["right_inner_finger"]),
        }
    )
    finger_opposition = RewTerm(
        func=mdp.finger_opposition_reward,
        weight=3.0,
        params={
            'target_asset_cfg': SceneEntityCfg("cube"),
            'left_finger_cfg': SceneEntityCfg("robot", body_names=["left_inner_finger"]),
            'right_finger_cfg': SceneEntityCfg("robot", body_names=["right_inner_finger"]),
        }
    )

    finger_height_alignment = RewTerm(
        func=mdp.finger_height_alignment_reward,
        weight=3.0,
        params={
            'left_finger_cfg': SceneEntityCfg("robot", body_names=["left_inner_finger"]),
            'right_finger_cfg': SceneEntityCfg("robot", body_names=["right_inner_finger"]),
            'target_asset_cfg': SceneEntityCfg("cube"),
            'std_height': CUBE_LENGTH/4, 
        }
    )

    contact_grasp = RewTerm(
        func=mdp.contact_grasp_reward, # increased from 0.0
        weight=5.0,
        params={'force_scale': 10.0, "sensor1_cfg": SceneEntityCfg("left_finger_contact_sensor"), "sensor2_cfg": SceneEntityCfg("right_finger_contact_sensor")}
    )
    
    wrist_outside_normal_to_target = RewTerm(
        func=mdp.wrist_outside_normal_to_target_reward,
        weight=3.0,
        params={
            "ee_link_cfg": SceneEntityCfg("robot", body_names=[EE_LINK_NAME]),
            "target_asset_cfg": SceneEntityCfg("cube"),
            "std_angle": 0.5,
        }
    )
    finger_line_horizontal = RewTerm(
        func=mdp.finger_line_horizontal_reward,
        weight=3.0,
        params={
            "left_finger_cfg": SceneEntityCfg("robot", body_names=["left_inner_finger"]),
            "right_finger_cfg": SceneEntityCfg("robot", body_names=["right_inner_finger"]),
        }
    )
    finger_closure = RewTerm(
        func=mdp.finger_closure_reward,
        weight=5.0,
        params={
            'left_finger_cfg': SceneEntityCfg("robot", body_names=["left_inner_finger"]),
            'right_finger_cfg': SceneEntityCfg("robot", body_names=["right_inner_finger"]),
            'target_width': CUBE_LENGTH,
            'activation_dist': CUBE_LENGTH*0.75,
            'std_gap': CUBE_LENGTH * 1.5
        }
    )
    # gripper_grasp_cube_reward = RewTerm(
    #     func=mdp.gripper_grasp_cube_reward,
    #     weight=0.5,
    #     params={
    #         'std_dist': STD_DIST,
    #         'std_grasp': 0.03,
    #         'target_asset_cfg': SceneEntityCfg("cube"),
    #         'dist_tolerance': DIST_TOLERANCE,
    #         'grasp_success_threshold': 0.2,
    #         'grasp_success_reward': 50.0,
    #     }
    # )
    # finger_gap_reward = RewTerm(
    #     func=mdp.finger_gap_reward,
    #     weight=1.0,
    #     params={
    #         'cube_length': CUBE_LEGNTH,
    #         'gap_far_offset': 0.06,
    #         'gap_near_offset': -0.005, 
    #         'gap_std': 0.03,
    #         'dist_far': 0.20,
    #         'dist_near': 0.05,
    #     }
    # )

    # Increase it in the CurriculumCfg
    cube_command_dist = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.0,
        params={
                # "std_dist": STD_DIST_MOVE,
                "std_dist": STD_DIST,
                "asset_cfg": SceneEntityCfg("cube"),
                "command_name": "move_target",
                'left_finger_cfg': SceneEntityCfg("robot", body_names=["left_inner_finger"]),
                'right_finger_cfg': SceneEntityCfg("robot", body_names=["right_inner_finger"]),
                }
    )
    cube_move_towards_command = RewTerm(
        func=mdp.asset_vel_to_command,
        weight=0.0,
        params={
                "asset_cfg": SceneEntityCfg("cube"),
                "command_name": "move_target",
                "cube_length": CUBE_LENGTH
                }
    )

    action_rate = RewTerm(
        func=mdp.action_rate_l2, 
        weight=-0.001
    )
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg("robot")}, # exclude mimic joints
    )
    # joint_vel = RewTerm(
    #     func=mdp.joint_vel_l2,
    #     weight=-0.01,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot",
    #                                     joint_ids=[0, 1, 2, 3, 4, 5, 6])}, # exclude mimic joints
    # )
    # gripper_effort_penalty = RewTerm(
    #     func=mdp.joint_torques_l2,
    #     weight=-0.001,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_ids=[6])
    #     }
    # )
    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-5.0,
    )


def joint_vel_too_high(env, threshold: float, asset_cfg: SceneEntityCfg):
    """Termination if any joint velocity exceeds the threshold."""
    asset = env.scene[asset_cfg.name]
    joint_vels = asset.data.joint_vel[:, asset_cfg.joint_ids]
    mask = torch.any(torch.abs(joint_vels) > threshold, dim=1)
    # Periodic debug: every 1000 steps print how many envs exceed the threshold
    if env.unwrapped.common_step_counter % 1000 == 0:
        count = int(mask.sum().item())
        if count:
            print(f"[DEBUG] joint_vel_too_high - count: {count}/{env.num_envs}, threshold: {threshold}, step: {env.unwrapped.common_step_counter}")
    return mask

def cube_distance_too_far(
    env,
    robot_cfg: SceneEntityCfg,
    cube_cfg: SceneEntityCfg,
    max_distance: float = 2.0,
) -> torch.Tensor:
    # Get robot base position
    robot_asset = env.scene[robot_cfg.name]
    robot_body_id = robot_asset.find_bodies(robot_cfg.body_names[0])[0]
    robot_pos = robot_asset.data.body_pos_w[:, robot_body_id, :3].squeeze(1)
    
    # Get cube position
    cube_asset = env.scene[cube_cfg.name]
    cube_pos = cube_asset.data.root_pos_w[:, :3]
    
    # Calculate distance
    distance = torch.norm(cube_pos - robot_pos, dim=1)
    
    # Return termination mask
    too_far = distance > max_distance
    
    # Debug print
    if env.unwrapped.common_step_counter % 10000 == 0 and too_far.any():
        print(f"[DEBUG] Cube too far - "
              f"count: {too_far.sum().item()}/{env.num_envs}, "
              f"max distance: {distance.max().item():.2f}m, "
              f"mean distance: {distance.mean().item():.2f}m")
    
    return too_far

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    joint_vel_limit = DoneTerm(
        func=joint_vel_too_high, 
        params={
            "threshold": 10.0,
            "asset_cfg": SceneEntityCfg("robot", joint_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        }
    )
    cube_too_far = DoneTerm(
        func=cube_distance_too_far,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=[BASE_LINK_NAME]),
            "cube_cfg": SceneEntityCfg("cube"),
            "max_distance": ENV_SPACING,
        }
    )


@configclass
class EventCfg:
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.8, 1.2),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_cube_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.35, 0.5),
                "y": (0.3, 0.5),
                "z": (CUBE_LENGTH/2 + 0.001, CUBE_LENGTH/2 + 0.001),  # Slightly above the ground to avoid initial penetration
            },
            'velocity_range': {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            "asset_cfg": SceneEntityCfg("cube"),
        },
    )

# def override_param(env, env_ids, data, value, num_steps):
#     cur_step = env.unwrapped.common_step_counter
#     # find the first num_steps smaller than cur_step and get the corresponding value
#     updated = False
#     for idx, step in enumerate(num_steps):
#         if cur_step >= step:
#             new_value = value[idx]
#             updated = True
#         else:
#             break
#     if updated:
#         return new_value
#     return mdp.modify_term_cfg.NO_CHANGE

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP"""
    # action_rate = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.005, "num_steps": 150000}
    # )
    # grasp_weight_increase = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "gripper_grasp_cube_reward", "weight": 1.5, "num_steps": 100000}
    # )
    # grasp_weight_final = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "gripper_grasp_cube_reward", "weight": 2.0, "num_steps": 300000}
    # )

    # tighten_threshold = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "rewards.gripper_grasp_cube_reward.params.grasp_success_threshold",
    #         "modify_params": {"value": [0.4, 0.5], "num_steps": [200000, 300000]},
    #         "modify_fn": override_param,
    #     }
    # )
    # closure_schedule = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "finger_closure", "weight": 5.0, "num_steps": 100000}
    # )
    contact_schedule0 = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "contact_grasp", "weight": 7.0, "num_steps": 500000}
    )
    contact_schedule1= CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "contact_grasp", "weight": 8.0, "num_steps": 600000}
    )
    contact_schedule2= CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "contact_grasp", "weight": 9.0, "num_steps": 700000}
    )
    contact_schedule3= CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "contact_grasp", "weight": 10.0, "num_steps": 800000}
    )
    contact_schedule4= CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "contact_grasp", "weight": 11.0, "num_steps": 900000}
    )
    contact_schedule5 = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "contact_grasp", "weight": 12.0, "num_steps": 1000000}
    )
    increase_move_reward0 = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "cube_command_dist",
            "weight": 1.0,          
            "num_steps": 500000
        }
    )
    increase_move_reward1 = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "cube_command_dist",
            "weight": 3.0,          
            "num_steps": 650000     
        }
    )
    increase_move_reward2 = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "cube_command_dist",
            "weight": 5.0,          
            "num_steps": 800000     
        }
    )
    increase_move_reward3 = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "cube_command_dist",
            "weight": 7.0,          
            "num_steps": 1000000     
        }
    )
    increase_move_reward4 = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "cube_command_dist",
            "weight": 8.0,          
            "num_steps": 1200000     
        }
    )
    increase_move_reward5 = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "cube_command_dist",
            "weight": 10.0,          
            "num_steps": 1500000     
        }
    )
    cube_move_towards_command0 = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "cube_move_towards_command",
            "weight": 1.0,          
            "num_steps": 500000
        }
    )
    cube_move_towards_command1 = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "cube_move_towards_command",
            "weight": 3.0,          
            "num_steps": 650000     
        }
    )
    cube_move_towards_command2 = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "cube_move_towards_command",
            "weight": 5.0,          
            "num_steps": 800000     
        }
    )
    cube_move_towards_command3 = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "cube_move_towards_command",
            "weight": 7.0,          
            "num_steps": 1000000     
        }
    )
    cube_move_towards_command4 = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "cube_move_towards_command",
            "weight": 8.0,          
            "num_steps": 1200000     
        }
    )
    cube_move_towards_command5 = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "cube_move_towards_command",
            "weight": 10.0,          
            "num_steps": 1400000     
        }
    )
    cube_move_towards_command6 = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "cube_move_towards_command",
            "weight": 12.0,          
            "num_steps": 1600000     
        }
    )
    cube_move_towards_command6 = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "cube_move_towards_command",
            "weight": 14.0,          
            "num_steps": 1800000     
        }
    )
    cube_move_towards_command6 = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "cube_move_towards_command",
            "weight": 16.0,          
            "num_steps": 2000000     
        }
    )
    cube_move_towards_command6 = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "cube_move_towards_command",
            "weight": 18.0,          
            "num_steps": 2200000     
        }
    )
    cube_move_towards_command6 = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "cube_move_towards_command",
            "weight": 200,          
            "num_steps": 2400000     
        }
    )
    action_rate_0 = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.003, "num_steps": 300000}
    )
    action_rate_1 = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.006, "num_steps": 1000000}
    )
    action_rate_2 = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.008, "num_steps": 1500000}
    )



##
# Environment configuration
##


@configclass
class ReachcubepickEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: ReachcubepickSceneCfg = ReachcubepickSceneCfg(num_envs=1000, env_spacing=ENV_SPACING)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    events = EventCfg()
    curriculum = CurriculumCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = EPISODE_LENGTH_S
        self.viewer.eye = (3.5, 3.5, 3.5)
        self.sim.dt = 1.0 / 120.0

@configclass
class ReachcubepickEnvCfg_PLAY(ReachcubepickEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 5
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
