
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

# FILE_NAME = "assets/UR-with-gripper.usd"
# FILE_NAME = "assets/UR-with-gripper_from_course.usd"
# FILE_NAME = "assets/UR-with-gripper_robot_assembler.usd"
# FILE_NAME = "assets/UR-with-gripper_robot_assembler0228.usd"
# BASE_LINK_NAME = "base_link"
EE_LINK_NAME = "ee_link"
# FILE_NAME = "assets/UR10e-with-gripper.usd" # almost OK, but gripper shakes
# FILE_NAME = "assets/UR10e-with-gripper-stiffness.usd"

# velocity_iterations to 32, arm+elbow stiffness/damping to 2000/100, wrist stiffness/damping to 1000/100, attach friction material to finger, finger stiffness/damping to 2000/200
FILE_NAME = "assets/UR10e-with-gripper-stiffness2000.usd" 
BASE_LINK_NAME = "world"
EE_LINK_NAME = "robotiq_arg2f_base_link"
ROBOT_PRIM_NAME = "ur_gripper"
GRIPPER_PRIM_NAME = "robotiq_2f_140"
UR_PATH = os.path.abspath(os.path.join(os.path.abspath(__file__),'../../../../../../../', FILE_NAME))
UR_GRIPPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=UR_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=10.0,
            max_angular_velocity=5.0,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=32
        )
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.0,
            "elbow_joint": 1.5,
            "wrist_1_joint": -1.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
        },
        # joint_pos={
        #     "shoulder_pan_joint": 0.0,
        #     "shoulder_lift_joint": 0.0,
        #     "elbow_joint": 0.0,
        #     "wrist_1_joint": 0.0,
        #     "wrist_2_joint": 0.0,
        #     "wrist_3_joint": 0.0,
        # },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
            ],
            velocity_limit_sim=3.14,
            effort_limit_sim=330.0,
            stiffness=2000.0,
            damping=100.0,
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=[
                "elbow_joint",
            ],
            velocity_limit_sim=3.14,
            effort_limit_sim=150.0,
            stiffness=2000.0,
            damping=100.0,
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=[
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            velocity_limit_sim=3.14,
            effort_limit_sim=56.0,
            stiffness=1000.0,
            damping=100.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[
                "finger_joint",
                # 'right_outer_knuckle_joint', 
                # 'left_outer_finger_joint', 
                # 'right_outer_finger_joint', 
                # 'left_inner_finger_joint', 
                # 'right_inner_finger_joint',
                # 'left_inner_finger_pad_joint', 
                # 'right_inner_finger_pad_joint'
            ],
            velocity_limit_sim=1.0,
            effort_limit_sim=100.0,
            stiffness=2000.0,
            damping=200.0,
        ),
    }
)