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

from .ur_gripper import UR_GRIPPER_CFG, UR_PATH, BASE_LINK_NAME, EE_LINK_NAME
import isaaclab.sim.schemas
import carb.settings
from pxr import Usd

def get_random_translation():
    x = random.uniform(0.3, 0.6)
    y = random.uniform(0.1, 0.2)
    z = CUBE_LEGNTH/2 + 0.001  # Slightly above the ground to avoid initial penetration
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
CUBE_LEGNTH = 0.08
DIST_TOLERANCE = CUBE_LEGNTH/5
GRASP_TOLERANCE = CUBE_LEGNTH/10
CUBE_MASS = 0.5
unit_scale = read_meters_per_unit_from_usd(UR_PATH)
INNER_FINGER_SIZE = [unit_scale*0.0655, 0, 0] # https://blog.robotiq.com/hubfs/support-files/2F-85_2F-140_UR_PDF_20240402.pdf
MIN_FINGER_GAP = 0.01
MAX_FINGER_GAP = 0.14
EPISODE_LENGTH_S = 6.0

@configclass
class ReachcubepickSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot = UR_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=CuboidCfg(
            size=(CUBE_LEGNTH, CUBE_LEGNTH, CUBE_LEGNTH),
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
    # finger_contact_sensor = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/ee_link/left_inner_finger",
    #     update_period=0.01,
    #     history_length=3,
    #     filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"]
    # )

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

        # Action and Command
        actions = ObsTerm(func=mdp.last_action)
        
        # Cube observations
        cube_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("cube")}) # TODO: should use pos relative to robot base
        cube_quat = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("cube")})

        ee_pos = ObsTerm(func=mdp.body_pose_w, params={"asset_cfg": SceneEntityCfg("robot", body_names=[EE_LINK_NAME])})
        # fingertip_gap = ObsTerm(func=get_left_right_fingertip_gap)
        finger_gap_native = ObsTerm(func=mdp.inner_finger_gap_native, params={
            "left_finger_cfg": SceneEntityCfg("robot", body_names=["left_inner_finger"]),
            "right_finger_cfg": SceneEntityCfg("robot", body_names=["right_inner_finger"]),
        })
        # cube_fingertip_mid_diff = ObsTerm(func=mdp.fingertip_midpoint_to_target_vector, params={
        #     "target_asset_cfg": SceneEntityCfg("cube")})
        finger_to_cube_native = ObsTerm(func=mdp.inner_finger_midpoint_to_target_native, params={
            "left_finger_cfg": SceneEntityCfg("robot", body_names=["left_inner_finger"]),
            "right_finger_cfg": SceneEntityCfg("robot", body_names=["right_inner_finger"]),
            "target_asset_cfg": SceneEntityCfg("cube")
        })
        # move_target_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "move_target"})
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    arm_action: ActionTerm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
        scale=1,
        use_default_offset=True,
        debug_vis=True
    )
    gripper_action: ActionTerm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        scale=1,
        use_default_offset=True,
        debug_vis=True
    )

@configclass
class CommandsCfg:
    move_target = mdp.UniformPoseCommandCfg(
        asset_name="robot", # target is based on the robot root
        body_name=BASE_LINK_NAME,
        resampling_time_range=(EPISODE_LENGTH_S, EPISODE_LENGTH_S),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.7),
            pos_y=(-0.2, 0.2),
            pos_z=(0, 0),
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
    #         'std_dist': 0.15,
    #         'target_asset_cfg': SceneEntityCfg("cube"),
    #     }
    # )
    gripper_cube_dist_reward_native = RewTerm(
        func=mdp.native_finger_midpoint_to_target_distance_reward,
        weight=3.0,
        params={
            'std_dist': 0.15,
            'left_finger_cfg': SceneEntityCfg("robot", body_names=["left_inner_finger"]),
            'right_finger_cfg': SceneEntityCfg("robot", body_names=["right_inner_finger"]),
            'target_asset_cfg': SceneEntityCfg("cube"),
        }
    )

    finger_grasp_reward_native = RewTerm(
        func=mdp.native_finger_grasp_reward,
        weight=3.0,
        params={
            'std_dist': 0.15,
            'left_finger_cfg': SceneEntityCfg("robot", body_names=["left_inner_finger"]),
            'right_finger_cfg': SceneEntityCfg("robot", body_names=["right_inner_finger"]),
            'target_asset_cfg': SceneEntityCfg("cube"),
        }
    )

    finger_gap_reward_native = RewTerm(
        func=mdp.native_finger_gap_reward,
        weight=3.0,
        params={
            'cube_length': CUBE_LEGNTH,
            'std_dist': 0.15,
            'left_finger_cfg': SceneEntityCfg("robot", body_names=["left_inner_finger"]),
            'right_finger_cfg': SceneEntityCfg("robot", body_names=["right_inner_finger"]),
            'target_asset_cfg': SceneEntityCfg("cube"),
        }
    )


    # gripper_grasp_cube_reward = RewTerm(
    #     func=mdp.gripper_grasp_cube_reward,
    #     weight=0.5,
    #     params={
    #         'std_dist': 0.15,
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
    # cube_move_position_tracking_tanh_sensor_activated = RewTerm(
    #     func=mdp.position_command_error_tanh,
    #     weight=5,
    #     params={"std": 0.2,
    #             "asset_cfg": SceneEntityCfg("cube"),
    #             "command_name": "move_target"}
    # )

    # contact_grasp_reward = RewTerm(
    #     func=mdp.contact_grasp_reward,
    #     weight=0.5, 
    #     params={
    #         'force_scale': 5.0
    #     }
    # )

    action_rate = RewTerm(
        func=mdp.action_rate_l2, 
        weight=-0.001
    )
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg("robot",
                                        joint_ids=[0, 1, 2, 3, 4, 5, 6])}, # exclude mimic joints
    )
    # joint_vel_gripper = RewTerm(
    #     func=mdp.joint_vel_l2,
    #     weight=-1.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_ids=[6, 7, 8, 9, 10, 11])
    #     },
    # )
    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-1.0,
    )


def joint_vel_too_high(env, threshold: float, asset_cfg: SceneEntityCfg):
    """Termination if any joint velocity exceeds the threshold."""
    asset = env.scene[asset_cfg.name]
    joint_vels = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.any(torch.abs(joint_vels) > threshold, dim=1)

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    joint_vel_limit = DoneTerm(
        func=joint_vel_too_high, 
        params={
            "threshold": 10.0,
            "asset_cfg": SceneEntityCfg("robot", joint_ids=[0, 1, 2, 3, 4, 5])
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
                "y": (-0.2, 0.3),
                "z": (CUBE_LEGNTH/2 + 0.001, CUBE_LEGNTH/2 + 0.001),  # Slightly above the ground to avoid initial penetration
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
