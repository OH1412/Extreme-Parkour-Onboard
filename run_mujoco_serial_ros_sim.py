"""Legacy MuJoCo serial runner split: simulation half.

This process owns MuJoCo only.  It publishes simulated observations and applies
motor commands received from ROS.  It does not load or run the policy.
"""

import argparse
import json
import os.path as osp
import time
from collections import OrderedDict

import mujoco
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from run_mujoco_heightmap_serial import (
    apply_task_config_overrides,
    build_mujoco_xml_with_terrain,
    generate_mujoco_terrain_mesh,
    generate_parkour_course,
    get_xml_hinge_joint_names,
)


MYBOT_V3_XML = osp.join(osp.dirname(osp.abspath(__file__)), "robots", "mybot_v3", "xml", "mybot_v3.xml")

DEPLOY_DOF_NAMES = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]


def quat_wxyz_to_roll_pitch_yaw(q):
    w, x, y, z = q
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def wrap_to_pi(angle):
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class MujocoRosBridge(Node):
    def __init__(self, args):
        super().__init__("mujoco_ros_bridge")
        self.args = args
        self.dof_names = DEPLOY_DOF_NAMES
        self.cfg = self._load_cfg(args.logdir)
        self.terrain = self._build_terrain()
        self.mujoco_xml = build_mujoco_xml_with_terrain(
            args.mujoco_xml,
            self.terrain,
            normalize_dynamics=args.normalize_mujoco_xml,
        )
        self.model = mujoco.MjModel.from_xml_path(self.mujoco_xml)
        self.data = mujoco.MjData(self.model)
        self.dof_names = DEPLOY_DOF_NAMES if bool(self.cfg.get("env", {}).get("reorder_dofs", True)) else get_xml_hinge_joint_names(self.model)
        self.joint_ids = self._lookup_joint_ids()
        self.actuator_ids = self._lookup_actuator_ids()
        self.qpos_adr = np.array([self.model.jnt_qposadr[jid] for jid in self.joint_ids], dtype=np.int32)
        self.qvel_adr = np.array([self.model.jnt_dofadr[jid] for jid in self.joint_ids], dtype=np.int32)
        self.default_dof_pos = np.array(
            [self.cfg["init_state"]["default_joint_angles"][name] for name in self.dof_names],
            dtype=np.float64,
        )
        self.height_points = self._init_height_points()
        heightmap_cfg = self.cfg.get("heightmap", {})
        self.height_clip_min = float(heightmap_cfg.get("clip_min", -1.0))
        self.height_clip_max = float(heightmap_cfg.get("clip_max", 1.0))
        self.raycast_bodyexclude = self._find_floating_base_body()
        self.robot_geom_ids = np.nonzero(self.model.geom_bodyid != 0)[0].astype(np.int32)
        self.raycast_geomgroup = np.ones(6, dtype=np.uint8)
        self.raycast_geomgroup[5] = 0
        self.low_cmd = None
        self.last_low_cmd_time = 0.0
        self.goal_yaw = np.zeros(3, dtype=np.float32)
        self.goals = None if self.terrain is None or "goals" not in self.terrain else np.asarray(self.terrain["goals"], dtype=np.float32)
        self.cur_goal_idx = 0
        self.reach_goal_timer = 0.0
        self.fake_wireless_keys = int(args.fake_wireless_keys)
        self.viewer = None
        self.last_heightmap_values = None
        self.last_heightmap_world_points = None

        if args.interface == "unitree":
            try:
                from unitree_go.msg import LowCmd, LowState, WirelessController
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "unitree_go is not available. Source the Unitree ROS2 workspace first, "
                    "or run this bridge with the default '--interface simple'."
                ) from exc
            self.low_cmd_msg_type = LowCmd
            self.low_state_msg_type = LowState
            self.wireless_msg_type = WirelessController
        else:
            self.low_cmd_msg_type = Float32MultiArray
            self.low_state_msg_type = Float32MultiArray
            self.wireless_msg_type = Float32MultiArray

        self.low_state_pub = self.create_publisher(self.low_state_msg_type, args.low_state_topic, 1)
        self.heightmap_pub = self.create_publisher(Float32MultiArray, args.heightmap_topic, 1)
        self.goal_yaw_pub = self.create_publisher(Float32MultiArray, args.goal_yaw_topic, 1)
        self.wireless_pub = self.create_publisher(self.wireless_msg_type, args.joy_stick_topic, 1)
        self.low_cmd_sub = self.create_subscription(self.low_cmd_msg_type, args.low_cmd_topic, self._low_cmd_callback, 1)

        self.reset()
        if args.render:
            import mujoco.viewer as mujoco_viewer

            self.viewer = mujoco_viewer.launch_passive(self.model, self.data)
        self.substeps = max(1, int(round(args.control_dt / self.model.opt.timestep)))
        self.timer = self.create_timer(args.control_dt, self.step)
        self.get_logger().info(
            "MuJoCo ROS bridge started: "
            f"interface={args.interface} "
            f"render={args.render} "
            f"control_dt={args.control_dt} "
            f"mujoco_timestep={self.model.opt.timestep} "
            f"substeps={self.substeps} "
            f"terrain={args.mujoco_terrain} "
            f"pub {args.low_state_topic}, {args.heightmap_topic}, {args.goal_yaw_topic}, {args.joy_stick_topic}; "
            f"sub {args.low_cmd_topic}"
        )

    def _build_terrain(self):
        if self.args.mujoco_terrain == "flat":
            return None
        if self.args.mujoco_terrain == "parkour":
            self.get_logger().info(
                f"Generating parkour box terrain difficulty={self.args.terrain_difficulty} seed={self.args.terrain_seed}"
            )
            return generate_parkour_course(
                self.cfg,
                seed=self.args.terrain_seed,
                difficulty=self.args.terrain_difficulty,
            )
        self.get_logger().info(
            f"Generating {self.args.mujoco_terrain} terrain difficulty={self.args.terrain_difficulty} "
            f"seed={self.args.terrain_seed} collision={self.args.terrain_collision}"
        )
        return generate_mujoco_terrain_mesh(
            self.cfg,
            terrain_kind=self.args.mujoco_terrain,
            seed=self.args.terrain_seed,
            difficulty=self.args.terrain_difficulty,
            collision_mode=self.args.terrain_collision,
            box_stride=self.args.terrain_box_stride,
            box_height_step=self.args.terrain_box_height_step,
        )

    def _load_cfg(self, logdir):
        if logdir:
            with open(osp.join(logdir, "config.json"), "r") as f:
                cfg = json.load(f, object_pairs_hook=OrderedDict)
            return apply_task_config_overrides(cfg, self.args.task_config)
        cfg = {
            "init_state": {
                "pos": [0.0, 0.0, 0.35],
                "rot": [0.0, 0.0, 0.0, 1.0],
                "default_joint_angles": {name: 0.0 for name in self.dof_names},
            },
            "terrain": {},
            "heightmap": {},
        }
        return apply_task_config_overrides(cfg, self.args.task_config)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        init_pos = self.cfg.get("init_state", {}).get("pos", [0.0, 0.0, 0.35])
        init_rot_xyzw = self.cfg.get("init_state", {}).get("rot", [0.0, 0.0, 0.0, 1.0])
        self.data.qpos[:3] = np.array(init_pos, dtype=np.float64)
        self.data.qpos[3:7] = np.array(
            [init_rot_xyzw[3], init_rot_xyzw[0], init_rot_xyzw[1], init_rot_xyzw[2]],
            dtype=np.float64,
        )
        self.data.qpos[self.qpos_adr] = self.default_dof_pos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def step(self):
        if self.viewer is not None and not self.viewer.is_running():
            self.get_logger().info("MuJoCo viewer closed; shutting down bridge.")
            rclpy.shutdown()
            return
        if not self._has_fresh_low_cmd():
            self.data.ctrl[:] = 0.0
            self._publish_observation_topics()
            self._sync_viewer()
            return
        self._apply_low_cmd()
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        self._publish_observation_topics()
        self._sync_viewer()

    def _has_fresh_low_cmd(self):
        if self.low_cmd is None:
            return False
        return time.monotonic() - self.last_low_cmd_time <= self.args.cmd_timeout

    def _publish_observation_topics(self):
        self._publish_low_state()
        self._publish_heightmap()
        self._update_goal_yaw()
        self._publish_goal_yaw()
        if self.args.publish_fake_wireless:
            self._publish_fake_wireless()

    def _sync_viewer(self):
        if self.viewer is not None:
            self._update_viewer_markers()
            self.viewer.sync()

    def _low_cmd_callback(self, msg):
        self.low_cmd = msg
        self.last_low_cmd_time = time.monotonic()

    def _apply_low_cmd(self):
        if self.low_cmd is None:
            self.data.ctrl[:] = 0.0
            return
        if time.monotonic() - self.last_low_cmd_time > self.args.cmd_timeout:
            self.data.ctrl[:] = 0.0
            return
        q = self.data.qpos[self.qpos_adr]
        dq = self.data.qvel[self.qvel_adr]
        if self.args.interface == "unitree":
            for dof_idx, actuator_id in enumerate(self.actuator_ids):
                cmd = self.low_cmd.motor_cmd[dof_idx]
                tau = cmd.kp * (cmd.q - q[dof_idx]) + cmd.kd * (cmd.dq - dq[dof_idx]) + cmd.tau
                self.data.ctrl[actuator_id] = float(np.clip(tau, -self.args.torque_limit, self.args.torque_limit))
            return

        cmd_data = np.asarray(self.low_cmd.data, dtype=np.float64)
        expected = 5 * len(self.dof_names)
        if cmd_data.size != expected:
            self.get_logger().warn(
                f"Simple low_cmd expects {expected} floats [q,dq,kp,kd,tau], got {cmd_data.size}.",
                throttle_duration_sec=1,
            )
            self.data.ctrl[:] = 0.0
            return
        n = len(self.dof_names)
        q_cmd = cmd_data[0:n]
        dq_cmd = cmd_data[n : 2 * n]
        kp = cmd_data[2 * n : 3 * n]
        kd = cmd_data[3 * n : 4 * n]
        tau_ff = cmd_data[4 * n : 5 * n]
        tau = kp * (q_cmd - q) + kd * (dq_cmd - dq) + tau_ff
        tau = np.clip(tau, -self.args.torque_limit, self.args.torque_limit)
        for dof_idx, actuator_id in enumerate(self.actuator_ids):
            self.data.ctrl[actuator_id] = float(tau[dof_idx])

    def _publish_low_state(self):
        quat = self.data.qpos[3:7].copy()
        ang_vel = self.data.qvel[3:6].copy()
        contacts = self._foot_contacts()

        if self.args.interface == "unitree":
            msg = self.low_state_msg_type()
            msg.imu_state.quaternion[0] = float(quat[0])
            msg.imu_state.quaternion[1] = float(quat[1])
            msg.imu_state.quaternion[2] = float(quat[2])
            msg.imu_state.quaternion[3] = float(quat[3])
            for i in range(3):
                msg.imu_state.gyroscope[i] = float(ang_vel[i])
            for i in range(len(self.dof_names)):
                msg.motor_state[i].q = float(self.data.qpos[self.qpos_adr[i]])
                msg.motor_state[i].dq = float(self.data.qvel[self.qvel_adr[i]])
            for i in range(4):
                msg.foot_force[i] = float(50.0 if contacts[i] else 0.0)
        else:
            msg = Float32MultiArray()
            msg.data = np.concatenate(
                [
                    quat.astype(np.float32),
                    ang_vel.astype(np.float32),
                    self.data.qpos[self.qpos_adr].astype(np.float32),
                    self.data.qvel[self.qvel_adr].astype(np.float32),
                    contacts.astype(np.float32),
                ]
            ).tolist()
        self.low_state_pub.publish(msg)

    def _publish_heightmap(self):
        base_pos, yaw = self._base_pos_yaw()
        msg = Float32MultiArray()
        msg.data = self._sample_heightmap(base_pos, yaw).astype(np.float32).tolist()
        self.heightmap_pub.publish(msg)

    def _publish_goal_yaw(self):
        msg = Float32MultiArray()
        msg.data = self.goal_yaw.tolist()
        self.goal_yaw_pub.publish(msg)

    def _update_goal_yaw(self):
        if self.goals is None or len(self.goals) == 0:
            self.goal_yaw[:] = 0.0
            return
        base_pos, yaw = self._base_pos_yaw()
        threshold = float(self.cfg.get("env", {}).get("next_goal_threshold", 0.2))
        reach_delay = float(self.cfg.get("env", {}).get("reach_goal_delay", 0.1))
        goal_idx = min(self.cur_goal_idx, len(self.goals) - 1)
        cur_goal = self.goals[goal_idx]
        if np.linalg.norm(base_pos[:2] - cur_goal[:2]) < threshold:
            self.reach_goal_timer += self.args.control_dt
            if self.reach_goal_timer > reach_delay and self.cur_goal_idx < len(self.goals) - 1:
                self.cur_goal_idx += 1
                self.reach_goal_timer = 0.0
        else:
            self.reach_goal_timer = 0.0

        goal_idx = min(self.cur_goal_idx, len(self.goals) - 1)
        next_idx = min(goal_idx + 1, len(self.goals) - 1)
        cur_goal = self.goals[goal_idx]
        next_goal = self.goals[next_idx]
        cur_vec = cur_goal[:2] - base_pos[:2]
        next_vec = next_goal[:2] - base_pos[:2]
        cur_yaw = np.arctan2(cur_vec[1], cur_vec[0])
        next_yaw = np.arctan2(next_vec[1], next_vec[0])
        self.goal_yaw[:] = [0.0, wrap_to_pi(cur_yaw - yaw), wrap_to_pi(next_yaw - yaw)]

    def _publish_fake_wireless(self):
        msg = self.wireless_msg_type()
        if self.args.interface == "unitree":
            msg.keys = self.fake_wireless_keys
        else:
            msg.data = [float(self.fake_wireless_keys)]
        self.wireless_pub.publish(msg)

    def _base_pos_yaw(self):
        base_pos = self.data.qpos[:3].copy().astype(np.float32)
        _, _, yaw = quat_wxyz_to_roll_pitch_yaw(self.data.qpos[3:7])
        return base_pos, float(yaw)

    def _sample_heightmap(self, base_pos, yaw):
        c = np.cos(yaw)
        s = np.sin(yaw)
        local_xy = self.height_points[:, :2]
        world_xy = np.empty_like(local_xy)
        world_xy[:, 0] = c * local_xy[:, 0] - s * local_xy[:, 1] + base_pos[0]
        world_xy[:, 1] = s * local_xy[:, 0] + c * local_xy[:, 1] + base_pos[1]
        measured = np.zeros(self.height_points.shape[0], dtype=np.float32)
        ray_origin = np.zeros(3, dtype=np.float64)
        ray_vec = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        geom_id = np.zeros(1, dtype=np.int32)
        ray_start_z = float(base_pos[2] + 5.0)
        old_groups = self.model.geom_group[self.robot_geom_ids].copy()
        self.model.geom_group[self.robot_geom_ids] = 5
        try:
            for i, xy in enumerate(world_xy):
                ray_origin[:] = [float(xy[0]), float(xy[1]), ray_start_z]
                dist = mujoco.mj_ray(
                    self.model,
                    self.data,
                    ray_origin,
                    ray_vec,
                    self.raycast_geomgroup,
                    1,
                    self.raycast_bodyexclude,
                    geom_id,
                )
                if dist >= 0.0:
                    measured[i] = np.float32(ray_start_z - dist)
        finally:
            self.model.geom_group[self.robot_geom_ids] = old_groups
        heights = np.clip(float(base_pos[2]) - 0.3 - measured, self.height_clip_min, self.height_clip_max)
        self.last_heightmap_values = heights.astype(np.float32).copy()
        self.last_heightmap_world_points = np.column_stack([world_xy, measured]).astype(np.float32)
        return heights

    def _update_viewer_markers(self):
        if not self.args.visualize_heightmap and not self.args.visualize_goals:
            return
        scene = self.viewer.user_scn
        scene.ngeom = 0
        mat = np.eye(3, dtype=np.float64).reshape(-1)
        if self.args.visualize_heightmap:
            self._add_heightmap_markers(scene, mat)
        if self.args.visualize_goals:
            self._add_goal_markers(scene, mat)

    def _add_heightmap_markers(self, scene, mat):
        if self.last_heightmap_world_points is None:
            return
        size = np.array([self.args.heightmap_marker_size] * 3, dtype=np.float64)
        points = self.last_heightmap_world_points
        values = self.last_heightmap_values
        for idx, point in enumerate(points):
            if scene.ngeom >= scene.maxgeom:
                break
            value = 0.0 if values is None else float(values[idx])
            normalized = (value - self.height_clip_min) / max(self.height_clip_max - self.height_clip_min, 1e-6)
            normalized = float(np.clip(normalized, 0.0, 1.0))
            rgba = np.array([normalized, 0.2, 1.0 - normalized, 1.0], dtype=np.float32)
            pos = np.array([point[0], point[1], point[2] + self.args.heightmap_marker_size], dtype=np.float64)
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                size,
                pos,
                mat,
                rgba,
            )
            scene.ngeom += 1

    def _add_goal_markers(self, scene, mat):
        if self.goals is None or len(self.goals) == 0:
            return
        size = np.array([self.args.goal_marker_size] * 3, dtype=np.float64)
        goal_idx = min(self.cur_goal_idx, len(self.goals) - 1)
        next_idx = min(goal_idx + 1, len(self.goals) - 1)
        for idx, goal in enumerate(self.goals):
            if scene.ngeom >= scene.maxgeom:
                break
            if idx < goal_idx:
                rgba = np.array([0.35, 0.35, 0.35, 0.35], dtype=np.float32)
            elif idx == goal_idx:
                rgba = np.array([0.1, 1.0, 0.2, 1.0], dtype=np.float32)
            elif idx == next_idx:
                rgba = np.array([1.0, 0.85, 0.1, 1.0], dtype=np.float32)
            else:
                rgba = np.array([0.1, 0.55, 1.0, 0.75], dtype=np.float32)
            pos = np.array([goal[0], goal[1], goal[2] + self.args.goal_marker_size], dtype=np.float64)
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                size * (1.35 if idx == goal_idx else 1.0),
                pos,
                mat,
                rgba,
            )
            scene.ngeom += 1

    def _foot_contacts(self):
        feet = ["FR", "FL", "RR", "RL"]
        contacts = np.zeros(4, dtype=bool)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or ""
            geom2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or ""
            pair = f"{geom1} {geom2}"
            for foot_idx, prefix in enumerate(feet):
                if prefix in pair:
                    contacts[foot_idx] = True
        return contacts

    def _init_height_points(self):
        terrain_cfg = self.cfg.get("terrain", {})
        x = np.asarray(
            terrain_cfg.get(
                "measured_points_x",
                [-0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2],
            ),
            dtype=np.float32,
        )
        y = np.asarray(
            terrain_cfg.get(
                "measured_points_y",
                [-0.75, -0.6, -0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.45, 0.6, 0.75],
            ),
            dtype=np.float32,
        )
        grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
        points = np.zeros((grid_x.size, 3), dtype=np.float32)
        points[:, 0] = grid_x.reshape(-1)
        points[:, 1] = grid_y.reshape(-1)
        return points

    def _find_floating_base_body(self):
        for jid in range(self.model.njnt):
            if self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                return int(self.model.jnt_bodyid[jid])
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "body")
        return int(body_id) if body_id >= 0 else -1

    def _lookup_joint_ids(self):
        joint_ids = []
        missing = []
        for name in self.dof_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                missing.append(name)
            else:
                joint_ids.append(jid)
        if missing:
            raise ValueError(f"MuJoCo model is missing joints: {missing}")
        return joint_ids

    def _lookup_actuator_ids(self):
        actuator_ids = []
        missing = []
        for name, jid in zip(self.dof_names, self.joint_ids):
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                for candidate in range(self.model.nu):
                    if self.model.actuator_trntype[candidate] == mujoco.mjtTrn.mjTRN_JOINT:
                        if self.model.actuator_trnid[candidate, 0] == jid:
                            aid = candidate
                            break
            if aid < 0:
                missing.append(name)
            else:
                actuator_ids.append(aid)
        if missing:
            raise ValueError(f"MuJoCo model has no joint actuators for: {missing}")
        return actuator_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mujoco_xml", type=str, default=MYBOT_V3_XML)
    parser.add_argument("--logdir", type=str, default=None, help="Optional config.json source for defaults and height points")
    parser.add_argument("--task_config", type=str, default="mybot_v3", choices=["mybot_v3", "traced"])
    parser.add_argument("--low_state_topic", type=str, default="/lowstate")
    parser.add_argument("--low_cmd_topic", type=str, default="/lowcmd")
    parser.add_argument("--joy_stick_topic", type=str, default="/wirelesscontroller")
    parser.add_argument("--heightmap_topic", type=str, default="/parkour/heightmap_points")
    parser.add_argument("--goal_yaw_topic", type=str, default="/parkour/goal_yaw")
    parser.add_argument("--interface", type=str, default="simple", choices=["simple", "unitree"])
    parser.add_argument(
        "--mujoco_terrain",
        type=str,
        default="flat",
        choices=["flat", "parkour", "parkour_hurdle", "parkour_flat", "parkour_step", "parkour_gap", "demo"],
    )
    parser.add_argument("--terrain_seed", type=int, default=1)
    parser.add_argument("--terrain_difficulty", type=float, default=-1.0)
    parser.add_argument("--terrain_collision", type=str, default="box", choices=["box", "mesh"])
    parser.add_argument("--terrain_box_stride", type=int, default=2)
    parser.add_argument("--terrain_box_height_step", type=float, default=0.02)
    parser.add_argument("--normalize_mujoco_xml", action="store_true", default=False)
    parser.add_argument("--visualize_heightmap", action="store_true", default=False)
    parser.add_argument("--heightmap_marker_size", type=float, default=0.025)
    parser.add_argument("--visualize_goals", action="store_true", default=False)
    parser.add_argument("--goal_marker_size", type=float, default=0.08)
    parser.add_argument("--control_dt", type=float, default=0.02)
    parser.add_argument("--cmd_timeout", type=float, default=0.2)
    parser.add_argument("--torque_limit", type=float, default=33.5)
    parser.add_argument("--fake_wireless_keys", type=int, default=0, help="Published WirelessController.keys value")
    parser.add_argument("--publish_fake_wireless", action="store_true", default=False)
    parser.add_argument("--render", action="store_true", default=False, help="Open a MuJoCo viewer window")
    args = parser.parse_args()

    rclpy.init()
    node = MujocoRosBridge(args)
    try:
        rclpy.spin(node)
    finally:
        if node.viewer is not None:
            node.viewer.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
