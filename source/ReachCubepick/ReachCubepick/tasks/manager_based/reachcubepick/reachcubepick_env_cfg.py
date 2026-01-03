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
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.sim.spawners.shapes import CuboidCfg
from . import mdp
from .ur_gripper import UR_GRIPPER_CFG
import isaaclab.sim.schemas
##
# Scene definition
##

ENV_SPACING = 2.5
CUBE_SIZE = 0.05
def get_random_translation():
    x = random.uniform(0.35, 0.7)
    y = random.uniform(0.1, 0.2)
    z = CUBE_SIZE/2 + 0.001  # Slightly above the ground to avoid initial penetration
    if random.random() < 0.5:
        y = -y

    return (x, y, z)

@configclass
class ReachcubepickSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot = UR_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            mass_props=sim_utils.schemas.MassPropertiesCfg(mass=0.1),
            rigid_props=sim_utils.schemas.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg()
        ),
        init_state = RigidObjectCfg.InitialStateCfg(pos=get_random_translation())
    )

##
# MDP settings
##

@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        # For moving the gripper to the cube pos, we needn't a pose command
        # pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        actions = ObsTerm(func=mdp.last_action)
        cube_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("cube")})
        lift_target_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "lift_target"})
        ee_rel_cube_pos = ObsTerm(
            func=mdp.position_target_asset_error_vector,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["ee_link"]),
                "target_asset_cfg": SceneEntityCfg("cube"),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01)
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
        scale=1,
        use_default_offset=True,
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

    lift_target = mdp.UniformPoseCommandCfg(
        asset_name="robot", # target is based on the robot root
        body_name="base_link",
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.7),
            pos_y=(-0.2, 0.2),
            pos_z=(0.2, 0.5),
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

    # For lifting the cube to the command pos
    cube_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("cube"), "command_name": "lift_target"},
    )

    end_effector_cube_position_tracking = RewTerm(
        func=mdp.position_target_asset_error,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["ee_link"]), "target_asset_cfg": SceneEntityCfg("cube")},
    )

    # action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)
    # joint_vel = RewTerm(
    #     func=mdp.joint_vel_l2,
    #     weight=-0.0001,
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    # )

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
                "x": (0.35, 0.7),
                "y": (-0.1, 0.2),
                "z": (CUBE_SIZE/2 + 0.001, CUBE_SIZE/2 + 0.001),  # Slightly above the ground to avoid initial penetration
            },
            'velocity_range': {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            "asset_cfg": SceneEntityCfg("cube"),
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP"""

    # action_rate = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.005, "num_steps": 4500}
    # )

    # joint_vel = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.001, "num_steps": 4500}
    # )



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
        self.episode_length_s = 3.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        self.sim.dt = 1.0 / 60.0

@configclass
class ReachcubepickEnvCfg_PLAY(ReachcubepickEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False