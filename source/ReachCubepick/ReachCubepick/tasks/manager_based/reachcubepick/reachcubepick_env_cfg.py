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
from .mdp.helper import cube_ee_relative_vel
from ....helpers.robotiq_fingertip_center_helper import get_left_right_fingertip_gap

from .ur_gripper import UR_GRIPPER_CFG, UR_PATH, BASE_LINK_NAME, EE_LINK_NAME, ROBOT_PRIM_NAME, GRIPPER_PRIM_NAME
import isaaclab.sim.schemas
import carb.settings
from pxr import Usd
from dataclasses import make_dataclass, field
from typing import *

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
LEFT_FINGER_PRIM_NAME = "left_inner_finger"
RIGHT_FINGER_PRIM_NAME = "right_inner_finger"
LEFT_FINGER_PRIM_PATH = "{ENV_REGEX_NS}"+f"/Robot/{ROBOT_PRIM_NAME}/{GRIPPER_PRIM_NAME}/left_inner_finger"
RIGHT_FINGER_PRIM_PATH = "{ENV_REGEX_NS}"+f"/Robot/{ROBOT_PRIM_NAME}/{GRIPPER_PRIM_NAME}/right_inner_finger"

# LEFT_FINGER_PRIM_NAME = "left_inner_finger_pad"
# RIGHT_FINGER_PRIM_NAME = "right_inner_finger_pad"
# LEFT_FINGER_PRIM_PATH = "{ENV_REGEX_NS}"+f"/Robot/{ROBOT_PRIM_NAME}/{GRIPPER_PRIM_NAME}/left_inner_finger/{LEFT_FINGER_PRIM_NAME}"
# RIGHT_FINGER_PRIM_PATH = "{ENV_REGEX_NS}"+f"/Robot/{ROBOT_PRIM_NAME}/{GRIPPER_PRIM_NAME}/right_inner_finger/{RIGHT_FINGER_PRIM_NAME}"

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
        prim_path=LEFT_FINGER_PRIM_PATH,
        update_period=0.01,
        history_length=3,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"]
    )
    right_finger_contact_sensor = ContactSensorCfg(
        prim_path=RIGHT_FINGER_PRIM_PATH,
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
        joint_effort = ObsTerm(func=mdp.joint_effort,
                               params={
                                   "asset_cfg": SceneEntityCfg("robot", joint_ids=[6])
                                   }
                            )
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
                "left_finger_cfg": SceneEntityCfg("robot", body_names=[LEFT_FINGER_PRIM_NAME]),
                "right_finger_cfg": SceneEntityCfg("robot", body_names=[RIGHT_FINGER_PRIM_NAME]),
            }
        )
        env_origin = ObsTerm(func=mdp.get_env_origin)
        finger_gap_native = ObsTerm(func=mdp.inner_finger_gap_minus_cube_length_native, params={
            "left_finger_cfg": SceneEntityCfg("robot", body_names=[LEFT_FINGER_PRIM_NAME]),
            "right_finger_cfg": SceneEntityCfg("robot", body_names=[RIGHT_FINGER_PRIM_NAME]),
            "cube_length": CUBE_LENGTH
        })
        # cube_fingertip_mid_diff = ObsTerm(func=mdp.fingertip_midpoint_to_target_vector, params={
        #     "target_asset_cfg": SceneEntityCfg("cube")})
        # finger_to_cube_vel_native = ObsTerm(func=mdp.inner_finger_midpoint_vel_to_target_native, params={
        #     "left_finger_cfg": SceneEntityCfg("robot", body_names=[LEFT_FINGER_PRIM_NAME]),
        #     "right_finger_cfg": SceneEntityCfg("robot", body_names=[RIGHT_FINGER_PRIM_NAME]),
        #     "target_asset_cfg": SceneEntityCfg("cube")
        # })
        each_finger_to_target_native = ObsTerm(func=mdp.each_finger_to_target_native, params={
            "left_finger_cfg": SceneEntityCfg("robot", body_names=[LEFT_FINGER_PRIM_NAME]),
            "right_finger_cfg": SceneEntityCfg("robot", body_names=[RIGHT_FINGER_PRIM_NAME]),
            "target_asset_cfg": SceneEntityCfg("cube")
        })
        finger_midpoint_to_target_native = ObsTerm(func=mdp.finger_midpoint_to_target_native, params={
            "left_finger_cfg": SceneEntityCfg("robot", body_names=[LEFT_FINGER_PRIM_NAME]),
            "right_finger_cfg": SceneEntityCfg("robot", body_names=[RIGHT_FINGER_PRIM_NAME]),
            "target_asset_cfg": SceneEntityCfg("cube")
        })
        finger_quat_native = ObsTerm(func=mdp.finger_quat_native, params={
            "left_finger_cfg": SceneEntityCfg("robot", body_names=[LEFT_FINGER_PRIM_NAME]),
            "right_finger_cfg": SceneEntityCfg("robot", body_names=[RIGHT_FINGER_PRIM_NAME]),
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
        #         "left_finger_cfg": SceneEntityCfg("robot", body_names=[LEFT_FINGER_PRIM_NAME]),
        #         "right_finger_cfg": SceneEntityCfg("robot", body_names=[RIGHT_FINGER_PRIM_NAME]),
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
        cube_ee_relative_vel = ObsTerm(
            func=cube_ee_relative_vel,
            params={"ee_link_name": EE_LINK_NAME}
        )

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
            pos_z=(CUBE_LENGTH/2, CUBE_LENGTH/2),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )

@configclass
class RewardsCfg:
    gripper_cube_dist = RewTerm(
        func=mdp.native_finger_midpoint_to_target_distance_reward,
        weight=5.0,
        params={
            'std_dist': STD_DIST,
            'left_finger_cfg': SceneEntityCfg("robot", body_names=[LEFT_FINGER_PRIM_NAME]),
            'right_finger_cfg': SceneEntityCfg("robot", body_names=[RIGHT_FINGER_PRIM_NAME]),
            'target_asset_cfg': SceneEntityCfg("cube"),
        }
    )

    finger_symmetry = RewTerm(
        func=mdp.finger_symmetry_reward,
        weight=3.0,
        params={
            'std_grasp': STD_GRASP,
            'target_asset_cfg': SceneEntityCfg("cube"),
            'left_finger_cfg': SceneEntityCfg("robot", body_names=[LEFT_FINGER_PRIM_NAME]),
            'right_finger_cfg': SceneEntityCfg("robot", body_names=[RIGHT_FINGER_PRIM_NAME]),
        }
    )
    finger_opposition = RewTerm(
        func=mdp.finger_opposition_reward,
        weight=3.0,
        params={
            'target_asset_cfg': SceneEntityCfg("cube"),
            'left_finger_cfg': SceneEntityCfg("robot", body_names=[LEFT_FINGER_PRIM_NAME]),
            'right_finger_cfg': SceneEntityCfg("robot", body_names=[RIGHT_FINGER_PRIM_NAME]),
        }
    )

    finger_height_alignment = RewTerm(
        func=mdp.finger_height_alignment_reward,
        weight=3.0,
        params={
            'left_finger_cfg': SceneEntityCfg("robot", body_names=[LEFT_FINGER_PRIM_NAME]),
            'right_finger_cfg': SceneEntityCfg("robot", body_names=[RIGHT_FINGER_PRIM_NAME]),
            'target_asset_cfg': SceneEntityCfg("cube"),
            'std_height': CUBE_LENGTH/4, 
        }
    )

    contact_grasp = RewTerm(
        func=mdp.contact_grasp_reward, # increased from 0.0
        weight=5.0,
        params={'force_scale': 5.0, "sensor1_cfg": SceneEntityCfg("left_finger_contact_sensor"), "sensor2_cfg": SceneEntityCfg("right_finger_contact_sensor")}
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
            "left_finger_cfg": SceneEntityCfg("robot", body_names=[LEFT_FINGER_PRIM_NAME]),
            "right_finger_cfg": SceneEntityCfg("robot", body_names=[RIGHT_FINGER_PRIM_NAME]),
        }
    )
    finger_closure = RewTerm(
        func=mdp.finger_closure_reward,
        weight=5.0,
        params={
            'left_finger_cfg': SceneEntityCfg("robot", body_names=[LEFT_FINGER_PRIM_NAME]),
            'right_finger_cfg': SceneEntityCfg("robot", body_names=[RIGHT_FINGER_PRIM_NAME]),
            'target_width': CUBE_LENGTH,
            'activation_dist': CUBE_LENGTH*0.75,
            'std_gap': CUBE_LENGTH * 1.5
        }
    )

    # Increase it in the CurriculumCfg
    cube_command_dist = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.0,
        params={
                # "std_dist": STD_DIST_MOVE,
                "std_dist": STD_DIST,
                "asset_cfg": SceneEntityCfg("cube"),
                "command_name": "move_target",
                'left_finger_cfg': SceneEntityCfg("robot", body_names=[LEFT_FINGER_PRIM_NAME]),
                'right_finger_cfg': SceneEntityCfg("robot", body_names=[RIGHT_FINGER_PRIM_NAME]),
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
    anti_slip_penalty = RewTerm(
        func=mdp.finger_slip_penalty,
        params={
            "ee_link_name": EE_LINK_NAME,
            "sensor1_cfg": SceneEntityCfg("left_finger_contact_sensor"),
            "sensor2_cfg": SceneEntityCfg("right_finger_contact_sensor"),
            "slip_vel_threshold": 0.04,
            "grasp_force_threshold": 1.5
        },
        weight=1.5,
    )
    
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




def build_curriculum_terms(term_name: str, weights: List[float], num_steps: List[int]) -> Dict[str, CurrTerm]:
    if len(weights) != len(num_steps):
        raise ValueError(f"Length mismatch for {term_name}")
    
    terms = {}
    for i, (steps, weight) in enumerate(sorted(zip(num_steps, weights), key=lambda x: x[0])):
        terms[f"{term_name}_{i}"] = CurrTerm(
            func=mdp.modify_reward_weight,
            params={"term_name": term_name, "weight": weight, "num_steps": steps}
        )
    return terms


def create_curriculum_cfg(terms_dict: Dict[str, CurrTerm]) -> type:
    def _make_factory(v):
        return lambda: v
    
    fields = [
        (name, CurrTerm, field(default_factory=_make_factory(value)))
        for name, value in terms_dict.items()
    ]
    return configclass(make_dataclass("CurriculumCfg", fields))


_curriculum_terms = {
    **build_curriculum_terms("contact_grasp", 
        weights=[7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0],
        num_steps=[150000, 200000, 250000, 300000, 350000, 400000, 450000, 550000, 650000, 750000]
    ),
    **build_curriculum_terms("cube_command_dist", 
        weights=[1.0, 3.0, 5.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
        num_steps=[550000, 600000, 700000, 800000, 900000, 1000000, 1100000, 1200000, 1300000, 1400000, 1500000]
    ),
    **build_curriculum_terms("cube_move_towards_command", 
        weights=[1.0, 3.0, 5.0, 7.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 28.0],
        num_steps=[600000, 650000, 750000, 850000, 950000, 1050000, 1150000, 1250000, 1350000, 1450000, 1550000]
    ),
    **build_curriculum_terms("action_rate", 
        weights=[-0.003, -0.006, -0.008],
        num_steps=[500000, 950000, 1550000]
    )
}

CurriculumCfg = create_curriculum_cfg(_curriculum_terms)


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
