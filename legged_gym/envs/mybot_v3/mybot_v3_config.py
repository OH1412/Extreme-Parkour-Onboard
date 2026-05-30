# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
# 关节顺序就是 URDF 里 12 个转动关节出现的顺序
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class MybotV3RoughCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        reorder_dofs = False
        include_foot_contacts = False  # 自研机器人无足端接触传感器
        n_proprio = LeggedRobotCfg.env.n_proprio - 4
        num_observations = (
            n_proprio
            + LeggedRobotCfg.env.n_scan
            + LeggedRobotCfg.env.history_len * n_proprio
            + LeggedRobotCfg.env.n_priv_latent
            + LeggedRobotCfg.env.n_priv
        )

    class heightmap:
        use_heightmap = True           # 启用LiDAR高程蒸馏
        n_points = 132                 # 高程采样点数
        update_interval = 5
        buffer_len = 3
        noise_std = 0.02               # LiDAR测量噪声
        horizontal_noise = 0.02
        clip_min = -1.0
        clip_max = 1.0

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.45]  # x,y,z [m]
        default_joint_angles = {  # target angles [rad] when action = 0.0
            "FL_hip_joint": 0.1,
            "RL_hip_joint": 0.1,
            "FR_hip_joint": -0.1,
            "RR_hip_joint": -0.1,

            "FL_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "FR_thigh_joint": 0.8,
            "RR_thigh_joint": 1.0,

            "FL_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        }

    class control(LeggedRobotCfg.control):
        control_type = "P"
        stiffness = {"joint": 35.0}  # [N*m/rad]
        damping = {"joint": 1}  # [N*m*s/rad]
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/mybot_v3/urdf/mybot_v3.urdf"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["body"]
        self_collisions = 1
        collapse_fixed_joints = False
        flip_visual_attachments = False

    class rewards:
        class scales:
            # tracking rewards
            tracking_goal_vel = 1.5
            tracking_yaw = 0.5
            # regularization rewards
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -1.
            dof_acc = -2.5e-7
            collision = -5.
            action_rate = -0.1
            delta_torques = -1.0e-7
            torques = -0.000001
            hip_pos = -2.0
            dof_error = -0.05
            feet_stumble = -1
            feet_edge = -1

        only_positive_rewards = True
        tracking_sigma = 0.2
        soft_dof_pos_limit = .9
        soft_dof_vel_limit = 1
        soft_torque_limit = 0.6
        base_height_target = 0.26
        max_contact_force = 40.


class MybotV3RoughCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class heightmap_encoder:
        if_heightmap = MybotV3RoughCfg.heightmap.use_heightmap
        n_points = MybotV3RoughCfg.heightmap.n_points
        n_proprio_student = LeggedRobotCfg.env.n_proprio - 4  # 53 - 4 = 49 (去掉foot_contacts)
        backbone_hidden_dims = [128, 64]
        backbone_output_dim = 32
        hidden_size = 512
        output_dim = 32
        learning_rate = 1.e-3
        num_steps_per_env = MybotV3RoughCfg.heightmap.update_interval * 24
        latent_loss_weight = 1.0
        action_loss_weight = 1.0
        yaw_loss_weight = 1.0

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "rough_mybot_v3"
