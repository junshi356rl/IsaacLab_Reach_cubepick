
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Universal Robots
Referencing: https://github.com/ros-industrial/universal_robot
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import sys
import os
from os import path

FILE_NAME = "assets/UR-with-gripper.usd"
# FILE_NAME = "assets/UR-with-gripper_from_course.usd"
UR_PATH = os.path.abspath(os.path.join(os.path.abspath(__file__),'../../../../../../../', FILE_NAME))
UR_GRIPPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=UR_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=10.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0
        )
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.712,
            "elbow_joint": 1.712,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            effort_limit=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[
                "finger_joint",
                # "right_outer_knuckle_joint",
                # "left_inner_knuckle_joint",
                # "right_inner_knuckle_joint",
                # "left_outer_finger_joint",
                # "right_outer_finger_joint",
                # "left_inner_finger_joint",
                # "right_inner_finger_joint",
                # "right_inner_finger_pad_joint",
                # "left_inner_finger_pad_joint",
            ],
            effort_limit_sim=200.0,
            stiffness=280,
            damping=28,
        ),
    }
)