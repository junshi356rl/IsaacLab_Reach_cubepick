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

from . import mdp
from ....helpers.robotiq_fingertip_center_helper import get_left_right_fingertip_gap

from .ur_gripper import UR_GRIPPER_CFG, UR_PATH
import isaaclab.sim.schemas
import carb.settings
from pxr import Usd

def get_random_translation():
    x = random.uniform(0.3, 0.6)
    y = random.uniform(0.1, 0.2)
    z = CUBE_SIZE/2 + 0.001  # Slightly above the ground to avoid initial penetration
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
CUBE_SIZE = 0.08
DIST_TOLERANCE = CUBE_SIZE/5
GRASP_TOLERANCE = CUBE_SIZE/10
CUBE_MASS = 3.0
unit_scale = read_meters_per_unit_from_usd(UR_PATH)
INNER_FINGER_SIZE = [unit_scale*0.0655, 0, 0] # https://blog.robotiq.com/hubfs/support-files/2F-85_2F-140_UR_PDF_20240402.pdf
MIN_FINGER_GAP = 0.01
MAX_FINGER_GAP = 0.14


@configclass
class ReachcubepickSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot = UR_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
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
    finger_contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ee_link/left_inner_finger",
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
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        # gripper_aperture = ObsTerm(
        #     func=mdp.position_target_asset_delta_vector_norm,
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot", body_names=["left_inner_finger"]),
        #         "target_asset_cfg": SceneEntityCfg("robot", body_names=["right_inner_finger"]),
        #     }
        # )

        # Action and Command
        actions = ObsTerm(func=mdp.last_action)
        # move_target_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "move_target"})
        # lift_target_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "lift_target"})
        
        # Cube observations
        cube_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("cube")}) # TODO: should use pos relative to robot base
        cube_quat = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("cube")})


        ee_pos = ObsTerm(func=mdp.body_pose_w, params={"asset_cfg": SceneEntityCfg("robot", body_names=["ee_link"])})
        fingertip_gap = ObsTerm(func=get_left_right_fingertip_gap)
        cube_fingertip_mid_diff = ObsTerm(func=mdp.fingertip_midpoint_to_target_vector, params={
            "target_asset_cfg": SceneEntityCfg("cube")})
        # cube_lin_vel = ObsTerm(
        #     func=mdp.root_lin_vel_w,
        #     params={"asset_cfg": SceneEntityCfg("cube")}
        # )  # 3D
        # cube_ang_vel = ObsTerm(
        #     func=mdp.root_ang_vel_w,
        #     params={"asset_cfg": SceneEntityCfg("cube")}
        # )  # 3D
        
        # End-effector to cube observations
        # ee_cube_pos = ObsTerm(
        #     func=mdp.position_target_asset_delta_vector,
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot", body_names=["ee_link"]),
        #         "target_asset_cfg": SceneEntityCfg("cube"),
        #     },
        #     noise=Unoise(n_min=-0.01, n_max=0.01)
        # )
        # ee_quat = ObsTerm(func=mdp.orientation_target_link,
        #                   params={"asset_cfg": SceneEntityCfg("robot", body_names=["ee_link"])},
        #                   noise=Unoise(n_min=-0.01, n_max=0.01))
        # finger_sensor_forces = ObsTerm(
        #     func=mdp.contact_sensor_forces,
        #     params={"sensor_cfg": SceneEntityCfg(name="finger_contact_sensor")},
        #     noise=Unoise(n_min=-0.01, n_max=0.01)
        # )
        # ee_to_cube_angle = ObsTerm(
        #     func=mdp.gripper_to_target_angle_obs,
        #     params={
        #         "left_finger_cfg": SceneEntityCfg("robot", body_names=["left_inner_finger"]),
        #         "right_finger_cfg": SceneEntityCfg("robot", body_names=["right_inner_finger"]),
        #         "target_asset_cfg": SceneEntityCfg("cube"),
        #     },
        #     noise=Unoise(n_min=-0.01, n_max=0.01)
        # )

                # left_inner_finger_rel_cube_pos = ObsTerm(
        #     func=mdp.position_target_asset_delta_vector,
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot", body_names=["left_inner_finger"]),
        #         "target_asset_cfg": SceneEntityCfg("cube"),
        #     },
        #     noise=Unoise(n_min=-0.01, n_max=0.01)
        # )
        # right_inner_finger_rel_cube_pos = ObsTerm(
        #     func=mdp.position_target_asset_delta_vector,
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot", body_names=["right_inner_finger"]),
        #         "target_asset_cfg": SceneEntityCfg("cube"),
        #     },
        #     noise=Unoise(n_min=-0.01, n_max=0.01)
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
        scale=1,
        use_default_offset=True,
        debug_vis=True
    )

    gripper_action: ActionTerm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        scale=1,
        use_default_offset=True,
        # offset={"finger_joint": -2.0},
        # clip={"finger_joint": (-3.0*CUBE_SIZE, 0.0)},
        debug_vis=True
    )


@configclass
class CommandsCfg:
    # ee_pose = mdp.UniformPoseCommandCfg(
    #     asset_name="robot",
    #     body_name="ee_link",
    #     resampling_time_range=(4.0, 4.0),
    #     debug_vis=True,
    #     ranges=mdp.UniformPoseCommandCfg.Ranges(
    #         pos_x=(0.35, 0.65),
    #         pos_y=(-0.2, 0.2),
    #         pos_z=(0.15, 0.5),
    #         roll=(0.0, 0.0),
    #         pitch=(math.pi / 2, math.pi / 2),
    #         yaw=(-3.14, 3.14),
    #     ),
    # )

    # lift_target = mdp.UniformPoseCommandCfg(
    #     asset_name="robot", # target is based on the robot root
    #     body_name="base_link",
    #     resampling_time_range=(4.0, 4.0),
    #     debug_vis=True,
    #     ranges=mdp.UniformPoseCommandCfg.Ranges(
    #         pos_x=(0.4, 0.7),
    #         pos_y=(-0.2, 0.2),
    #         pos_z=(0.2, 0.5),
    #         roll=(0.0, 0.0),
    #         pitch=(0.0, 0.0),
    #         yaw=(0.0, 0.0),
    #     ),
    # )

    move_target = mdp.UniformPoseCommandCfg(
        asset_name="robot", # target is based on the robot root
        body_name="base_link",
        resampling_time_range=(4.0, 4.0),
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
    # For moving the gripper to arbitrary position in the env
    # end_effector_orientation_tracking = RewTerm(
    #     func=mdp.orientation_command_error,
    #     weight=-0.1,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=["ee_link"]), "command_name": "ee_pose"},
    # )

    # end_effector_position_tracking = RewTerm(
    #     func=mdp.position_command_error,
    #     weight=-0.2,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=["ee_link"]), "command_name": "ee_pose"},
    # )
    # end_effector_position_tracking_fine_grained = RewTerm(
    #     func=mdp.position_command_error_tanh,
    #     weight=0.1,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=["ee_link"]), "std": 0.1, "command_name": "ee_pose"},
    # )

    # # For moving the gripper to the cube pos
    # end_effector_to_cube_position_tracking = RewTerm(
    #     func=mdp.position_target_asset_error,
    #     weight=-1.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=["ee_link"]), "target_asset_cfg": SceneEntityCfg("cube")},
    # )

    # # For lifting the cube to the command pos
    # cube_position_tracking = RewTerm(
    #     func=mdp.position_command_error,
    #     weight=-0.2,
    #     params={"asset_cfg": SceneEntityCfg("cube"), "command_name": "lift_target"},
    # )
    # cube_position_tracking_tanh = RewTerm(
    #     func=mdp.position_command_error_tanh,
    #     weight=0.5,
    #     params={"std": 0.1, "asset_cfg": SceneEntityCfg("cube"), "command_name": "lift_target"},
    # )

    # cube_move_position_tracking_tanh = RewTerm(
    #     func=mdp.position_command_error_tanh,
    #     weight=0.5,
    #     params={"std": 0.1, "asset_cfg": SceneEntityCfg("cube"), "command_name": "move_target"},
    # )  # the robot just pushes the cube

    # left_finger_cube_position_tracking = RewTerm(
    #     func=mdp.position_target_asset_error_tanh,
    #     weight=0.1,
    #     params={"std": 0.1, "asset_cfg": SceneEntityCfg("robot", body_names=["left_inner_finger"]), "target_asset_cfg": SceneEntityCfg("cube")},
    # )

    # right_finger_cube_position_tracking = RewTerm(
    #     func=mdp.position_target_asset_error_tanh,
    #     weight=0.1,
    #     params={"std": 0.1, "asset_cfg": SceneEntityCfg("robot", body_names=["right_inner_finger"]), "target_asset_cfg": SceneEntityCfg("cube")},
    # )

    # left_finger_center_cube_position_tracking = RewTerm(
    #     func=mdp.asset_center_position_target_asset_error_tanh,
    #     weight=0.1,
    #     params={"std": 0.1,
    #             "asset_cfg": SceneEntityCfg("robot", body_names=["left_inner_finger"]),
    #             "asset_size": INNER_FINGER_SIZE,
    #             "target_asset_cfg": SceneEntityCfg("cube")},
    # )

    # right_finger_cube_position_tracking = RewTerm(
    #     func=mdp.asset_center_position_target_asset_error_tanh,
    #     weight=0.1,
    #     params={"std": 0.1,
    #             "asset_cfg": SceneEntityCfg("robot", body_names=["right_inner_finger"]),
    #             "asset_size": INNER_FINGER_SIZE,
    #             "target_asset_cfg": SceneEntityCfg("cube")},
    # )

    # left_finger_activated = RewTerm(
    #     func=mdp.sensor_activated,
    #     weight=0.1,
    #     params={"sensor_cfg": SceneEntityCfg(
    #         name="finger_contact_sensor")
    #     }
    # )

    # contact_bonus = RewTerm(
    #     func=mdp.sensor_activated,
    #     params={"sensor_cfg": SceneEntityCfg(name="finger_contact_sensor")},
    #     weight=0.2,
    # )

    # cube_move_position_tracking_tanh_sensor_activated = RewTerm(
    #     func=mdp.position_command_error_tanh,
    #     weight=5,
    #     params={"std": 0.1,
    #             "asset_cfg": SceneEntityCfg("cube"),
    #             "command_name": "move_target"}
    # )

    # gripper_frontal_alignment = RewTerm(
    #     func=mdp.align_gripper_approach_direction_reward,
    #     weight=0.5,
    #     params={
    #         "std": 0.5,
    #         "left_finger_cfg": SceneEntityCfg("robot", body_names=["left_inner_finger"]),
    #         "right_finger_cfg": SceneEntityCfg("robot", body_names=["right_inner_finger"]),
    #         "target_asset_cfg": SceneEntityCfg("cube"),
    #     }
    # )

    # symmetric_finger_side_grasp = RewTerm(
    #     func=mdp.symmetric_finger_side_grasp_reward,
    #     weight=1.0,
    #     params={
    #         "std": 0.1,
    #         "ideal_gap": CUBE_SIZE,
    #         "finger_tip_offset_left": INNER_FINGER_SIZE,
    #         "finger_tip_offset_right": INNER_FINGER_SIZE,
    #         "left_finger_name": "left_inner_finger",
    #         "right_finger_name": "right_inner_finger",
    #         "min_gap": MIN_FINGER_GAP,
    #         "max_gap": MAX_FINGER_GAP,
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "target_asset_cfg": SceneEntityCfg("cube"),
    #     }
    # )

    # finger_mid_point_to_cube_position_tracking = RewTerm(
    #     func=mdp.two_finger_midpoint_to_target_asset_distance_reward,
    #     weight=0.5,
    #     params={
    #         "std": 0.1,
    #         "finger_tip_offset_left": INNER_FINGER_SIZE,
    #         "finger_tip_offset_right": INNER_FINGER_SIZE,
    #         "left_finger_name": "left_inner_finger",
    #         "right_finger_name": "right_inner_finger",
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "target_asset_cfg": SceneEntityCfg("cube"),
    #     }
    # )

    # TODO: Add a reward to adjust gripper parallel to the cube surface normal

    # action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)
    # joint_vel = RewTerm(
    #     func=mdp.joint_vel_l2,
    #     weight=-0.0001,
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    # )
    gripper_cube_dist_reward = RewTerm(
        func=mdp.gripper_target_dist_reward,
        weight=1.0,
        params={
            'std_dist': 0.05,
            'target_asset_cfg': SceneEntityCfg("cube"),
        }
    )

    gripper_grasp_cube_reward = RewTerm(
        func=mdp.gripper_grasp_cube_reward,
        weight=1.0,
        params={
            'std_dist': 0.05,
            'std_grasp': 0.03,
            'target_asset_cfg': SceneEntityCfg("cube"),
            'cube_length': CUBE_SIZE,
            'dist_tolerance': DIST_TOLERANCE,
            'grasp_tolerance': GRASP_TOLERANCE,
            'grasp_success_threshold': 0.15,
            'grasp_success_reward': 50.0,
        }
    )

    contact_grasp_reward = RewTerm(
        func=mdp.contact_grasp_reward,
        weight=0.5, 
        params={
            'force_scale': 5.0,
            'force_threshold': 0.5,
            'target_asset_cfg': SceneEntityCfg("cube"),
            'dist_tolerance': DIST_TOLERANCE
        }
    )

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class EventCfg:
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.75, 1.25),
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
                "z": (CUBE_SIZE/2 + 0.001, CUBE_SIZE/2 + 0.001),  # Slightly above the ground to avoid initial penetration
            },
            'velocity_range': {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            "asset_cfg": SceneEntityCfg("cube"),
        },
    )

def override_param(env, env_ids, data, value, num_steps):
    cur_step = env.unwrapped.common_step_counter
    # find the first num_steps smaller than cur_step and get the corresponding value
    updated = False
    for idx, step in enumerate(num_steps):
        if cur_step >= step:
            new_value = value[idx]
            updated = True
        else:
            break
    if updated:
        return new_value
    return mdp.modify_term_cfg.NO_CHANGE

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP"""

    # action_rate = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.005, "num_steps": 4500}
    # )

    # joint_vel = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.001, "num_steps": 4500}
    # )
    # enable_grasp = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={
    #         "term_name": "gripper_grasp_cube_reward",
    #         "weight": 1.0,
    #         "num_steps": 100000
    #     }
    # )

    tighten_threshold = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.gripper_grasp_cube_reward.params.grasp_success_threshold",
            "modify_params": {"value": [0.3, 0.4], "num_steps": [200000, 300000]},
            "modify_fn": override_param,
        }
    )


##
# Environment configuration
##


@configclass
class ReachcubepickEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: ReachcubepickSceneCfg = ReachcubepickSceneCfg(num_envs=2000, env_spacing=ENV_SPACING)
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
        self.episode_length_s = 8.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        self.sim.dt = 1.0 / 60.0

@configclass
class ReachcubepickEnvCfg_PLAY(ReachcubepickEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 5
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False