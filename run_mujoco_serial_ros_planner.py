"""Legacy MuJoCo serial runner split: planner/policy half.

This process keeps the keyboard, observation, policy, action scaling, and PD
target logic from run_mujoco_heightmap_serial.py.  It publishes the same
joint-space command format used by elmap-rl-controller deploy commit
77902c08a3feeef4f64745599a98e8dd1d9c8f29:

  std_msgs/Float32MultiArray [target_dof_pos(12), kp_joint(12), kd_joint(12)]
"""

import argparse
import json
import os.path as osp
import time
from collections import OrderedDict

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import torch

from run_mujoco_heightmap_serial import (
    DEFAULT_TEACHER_MODEL,
    ELMAP_SDK_CONFIG,
    TEACHER_NUM_HIST,
    TEACHER_NUM_PRIV_EXPLICIT,
    TEACHER_NUM_PRIV_LATENT,
    TEACHER_NUM_PROP,
    TEACHER_NUM_SCAN,
    KeyboardTeleopController,
    apply_task_config_overrides,
    get_xml_hinge_joint_names,
    resolve_model_path,
    wrap_to_pi,
)
from unitree_motor_sdk_python import PythonUnitreeMotorDriver


DEFAULT_DOF_NAMES = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]


def quat_wxyz_to_roll_pitch_yaw(q):
    w, x, y, z = q
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


class RosPolicyNode(Node):
    def __init__(self, args, cfg, policy, n_points, hidden_size, command_controller, sdk_motor=None):
        super().__init__("mujoco_heightmap_ros_policy")
        self.args = args
        self.cfg = cfg
        self.policy = policy
        self.device = args.device
        self.n_points = n_points
        self.command_controller = command_controller
        self.policy_type = args.policy_type
        self.sdk_motor = sdk_motor
        self.motor_backend = args.motor_backend
        self.joint_state_source = args.joint_state_source
        self.imu_source = args.imu_source
        self.dof_names = load_dof_names(args)
        self.num_dof = len(self.dof_names)
        self.n_proprio = TEACHER_NUM_PROP
        self.n_hist_len = TEACHER_NUM_HIST
        self.n_priv_explicit = TEACHER_NUM_PRIV_EXPLICIT
        self.n_priv_latent = TEACHER_NUM_PRIV_LATENT
        self.hidden = None if self.policy_type == "teacher" else torch.zeros(1, 1, hidden_size, device=self.device)
        self.proprio_history_buf = torch.zeros(1, self.n_hist_len, self.n_proprio, device=self.device)
        self.episode_length_buf = torch.zeros(1, device=self.device)
        self.raw_actions = torch.zeros(self.num_dof, device=self.device)
        self.actions = torch.zeros(self.num_dof, device=self.device)
        self.default_dof_pos = torch.tensor(
            [cfg["init_state"]["default_joint_angles"][name] for name in self.dof_names],
            device=self.device,
            dtype=torch.float32,
        )
        self.p_gains = self._gains_from_config("stiffness")
        self.d_gains = self._gains_from_config("damping")
        self.action_scale = float(cfg["control"]["action_scale"])
        self.clip_actions = float(cfg["normalization"]["clip_actions"])
        self.clip_observations = float(cfg["normalization"].get("clip_observations", 100.0))
        self.obs_scales = cfg["normalization"]["obs_scales"]
        self.include_foot_contacts = bool(cfg.get("env", {}).get("include_foot_contacts", True))
        if self.policy_type == "teacher":
            self.include_foot_contacts = True

        self.latest_lowstate = None
        self.latest_imu = None
        self.latest_heightmap = None
        self.latest_goal_yaw = torch.zeros(1, 3, device=self.device)
        self.last_cmd = {"mode": 0, "vx": 0.0, "vy": 0.0, "yaw": 0.0, "e_stop": False}
        self.loop_count = 0

        self.joint_cmd_topic = args.joint_cmd_topic
        self.joint_cmd_pub = (
            self.create_publisher(Float32MultiArray, self.joint_cmd_topic, 1)
            if self.motor_backend in ("ros", "both")
            else None
        )
        self.low_state_sub = None
        if self.uses_lowstate_joint() or self.uses_lowstate_imu():
            self.low_state_sub = self.create_subscription(Float32MultiArray, args.low_state_topic, self._low_state_callback, 1)
        self.imu_sub = None
        if self.uses_imu_topic():
            self.imu_sub = self.create_subscription(Float32MultiArray, args.imu_topic, self._imu_callback, 1)
        self.heightmap_sub = self.create_subscription(Float32MultiArray, args.heightmap_topic, self._heightmap_callback, 1)
        self.goal_yaw_sub = self.create_subscription(Float32MultiArray, args.goal_yaw_topic, self._goal_yaw_callback, 1)
        self.timer = self.create_timer(args.control_dt, self.step_policy)
        self.get_logger().info(
            f"ROS policy ready. dof_order={self.dof_names} control_dt={args.control_dt} "
            f"motor_backend={self.motor_backend} joint_state_source={self.joint_state_source} imu_source={self.imu_source} "
            f"lowstate={args.low_state_topic} heightmap={args.heightmap_topic} joint_cmd={self.joint_cmd_topic}"
        )

    def _gains_from_config(self, gain_key):
        gains = []
        gain_cfg = self.cfg["control"][gain_key]
        for name in self.dof_names:
            matched = False
            for key, value in gain_cfg.items():
                if key in name:
                    gains.append(float(value))
                    matched = True
                    break
            if not matched:
                raise KeyError(f"No {gain_key} gain matched joint {name}")
        return torch.tensor(gains, device=self.device, dtype=torch.float32)

    def _low_state_callback(self, msg):
        data = np.asarray(msg.data, dtype=np.float32)
        expected = 4 + 3 + self.num_dof + self.num_dof + 4
        if data.size != expected:
            self.get_logger().warn(f"lowstate expects {expected} floats, got {data.size}", throttle_duration_sec=1)
            return
        self.latest_lowstate = data.copy()

    def _imu_callback(self, msg):
        data = np.asarray(msg.data, dtype=np.float32)
        if data.size < 6:
            self.get_logger().warn(f"imu expects at least 6 floats [wx,wy,wz,gx,gy,gz], got {data.size}", throttle_duration_sec=1)
            return
        yaw = np.deg2rad(float(self.args.imu_yaw_correction_deg))
        c = np.cos(yaw)
        s = np.sin(yaw)
        wx, wy, wz = data[0:3]
        gx, gy, gz = data[3:6]
        self.latest_imu = np.array(
            [
                c * wx - s * wy,
                s * wx + c * wy,
                wz,
                c * gx - s * gy,
                s * gx + c * gy,
                gz,
            ],
            dtype=np.float32,
        )

    def _heightmap_callback(self, msg):
        data = torch.tensor(msg.data, device=self.device, dtype=torch.float32)
        if data.numel() != self.n_points:
            self.get_logger().warn(f"heightmap expects {self.n_points} floats, got {data.numel()}", throttle_duration_sec=1)
            return
        self.latest_heightmap = data.view(1, -1)

    def _goal_yaw_callback(self, msg):
        data = torch.tensor(msg.data, device=self.device, dtype=torch.float32)
        if data.numel() == 2:
            self.latest_goal_yaw = torch.tensor([[0.0, data[0].item(), data[1].item()]], device=self.device)
        elif data.numel() == 3:
            self.latest_goal_yaw = data.view(1, 3)
        else:
            self.get_logger().warn(f"goal_yaw expects 2 or 3 floats, got {data.numel()}", throttle_duration_sec=1)

    def step_policy(self):
        missing = []
        if self.latest_heightmap is None:
            missing.append(self.args.heightmap_topic)
        if self.uses_lowstate_joint() and self.latest_lowstate is None:
            missing.append(self.args.low_state_topic)
        if self.uses_lowstate_imu() and self.latest_lowstate is None and self.args.low_state_topic not in missing:
            missing.append(self.args.low_state_topic)
        if self.uses_imu_topic() and self.latest_imu is None:
            missing.append(self.args.imu_topic)
        if missing:
            self.get_logger().warn(f"Waiting for inputs: {', '.join(missing)}", throttle_duration_sec=1)
            return
        cmd = self.command_controller.get_latest()
        self.last_cmd = dict(cmd)
        if cmd["e_stop"] or int(cmd["mode"]) == 0:
            self.publish_zero_torque()
        elif int(cmd["mode"]) == 1 or int(cmd["mode"]) == 4:
            self.publish_pd(self.default_dof_pos, self.p_gains, self.d_gains)
        elif int(cmd["mode"]) == 3:
            self.publish_damping(0.2)
        elif int(cmd["mode"]) == 2:
            proprio = self.get_proprio(cmd)
            heightmap_points = self.latest_heightmap
            if self.policy_type == "teacher":
                obs = self.build_teacher_obs(proprio, heightmap_points)
                obs = self.clip_obs(obs)
                actions = self.policy(obs)
            else:
                obs = self.turn_obs(proprio, self.proprio_history_buf)
                obs = self.clip_obs(obs)
                actions, self.hidden = self.policy(obs, heightmap_points, self.hidden)
            self.send_action(actions)
        else:
            self.publish_pd(self.default_dof_pos, self.p_gains, self.d_gains)
        self.loop_count += 1

    def get_proprio(self, cmd):
        q_np, dq_np = self.current_joint_state()
        q = torch.tensor(q_np, device=self.device, dtype=torch.float32)
        dq = torch.tensor(dq_np, device=self.device, dtype=torch.float32)
        ang_vel_np, roll, pitch = self.current_imu_state()
        contacts_np = self.current_contacts()

        ang_vel = torch.tensor(ang_vel_np, device=self.device, dtype=torch.float32).view(1, 3)
        ang_vel = ang_vel * float(self.obs_scales["ang_vel"])
        imu = torch.tensor([[roll, pitch]], device=self.device, dtype=torch.float32)
        yaw_info = self.latest_goal_yaw.to(device=self.device, dtype=torch.float32)
        commands = torch.tensor([[0.0, 0.0, float(cmd["vx"])]], device=self.device)
        parkour_walk = torch.tensor(
            [[1.0, 0.0] if self.args.mode == "parkour" else [0.0, 1.0]],
            device=self.device,
            dtype=torch.float32,
        )
        dof_pos = (q.view(1, -1) - self.default_dof_pos.view(1, -1)) * float(self.obs_scales["dof_pos"])
        dof_vel = dq.view(1, -1) * float(self.obs_scales["dof_vel"])
        last_actions = self.raw_actions.view(1, -1)
        if self.include_foot_contacts:
            contacts = torch.tensor(contacts_np, device=self.device, dtype=torch.float32).view(1, 4)
            contact_obs = torch.where(contacts >= 0.5, torch.full_like(contacts, 0.5), torch.full_like(contacts, -0.5))
        else:
            contact_obs = torch.full((1, 4), -0.5, device=self.device, dtype=torch.float32)
        proprio = torch.cat([ang_vel, imu, yaw_info, commands, parkour_walk, dof_pos, dof_vel, last_actions, contact_obs], dim=-1)
        if proprio.shape[-1] != self.n_proprio:
            raise RuntimeError(f"Built proprio has {proprio.shape[-1]} dims, expected {self.n_proprio}.")
        self.proprio_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([proprio] * self.n_hist_len, dim=1),
            torch.cat([self.proprio_history_buf[:, 1:], proprio.unsqueeze(1)], dim=1),
        )
        self.episode_length_buf += 1
        return proprio

    def turn_obs(self, proprio, proprio_history):
        batch_size = proprio.shape[0]
        scan_zeros = torch.zeros(batch_size, self.n_points, device=self.device)
        priv_explicit_zeros = torch.zeros(batch_size, self.n_priv_explicit, device=self.device)
        priv_latent_zeros = torch.zeros(batch_size, self.n_priv_latent, device=self.device)
        return torch.cat(
            [proprio, scan_zeros, priv_explicit_zeros, priv_latent_zeros, proprio_history.view(batch_size, -1)],
            dim=-1,
        )

    def build_teacher_obs(self, proprio, heightmap_points):
        batch_size = proprio.shape[0]
        priv_explicit = torch.zeros(batch_size, self.n_priv_explicit, device=self.device)
        priv_latent = torch.zeros(batch_size, self.n_priv_latent, device=self.device)
        return torch.cat([proprio, heightmap_points, priv_explicit, priv_latent, self.proprio_history_buf.view(batch_size, -1)], dim=-1)

    def send_action(self, actions):
        hard_clip = self.clip_actions / self.action_scale
        raw_actions = actions.view(-1).to(self.device)
        clipped_actions = torch.clip(raw_actions, -hard_clip, hard_clip)
        self.raw_actions = raw_actions
        self.actions = clipped_actions
        target = clipped_actions * self.action_scale + self.default_dof_pos
        self.publish_pd(target, self.p_gains, self.d_gains)

    def publish_pd(self, q_target, kp, kd):
        q_np = q_target.detach().cpu().numpy().astype(np.float32)
        kp_np = kp.detach().cpu().numpy().astype(np.float32)
        kd_np = kd.detach().cpu().numpy().astype(np.float32)
        if self.motor_backend in ("ros", "both"):
            msg = Float32MultiArray()
            msg.data = np.concatenate([q_np, kp_np, kd_np]).tolist()
            self.joint_cmd_pub.publish(msg)
        if self.motor_backend in ("sdk", "both"):
            self.sdk_motor.send_commands(q_np.tolist(), kp_np.tolist(), kd_np.tolist())

    def publish_damping(self, kd):
        n = self.num_dof
        q_target = self.current_q_or_zeros()
        kp = np.zeros(n, dtype=np.float32)
        kd_arr = np.ones(n, dtype=np.float32) * float(kd)
        if self.motor_backend in ("ros", "both"):
            msg = Float32MultiArray()
            msg.data = np.concatenate([q_target, kp, kd_arr]).tolist()
            self.joint_cmd_pub.publish(msg)
        if self.motor_backend in ("sdk", "both"):
            self.sdk_motor.send_damping(float(kd))

    def publish_zero_torque(self):
        n = self.num_dof
        if self.motor_backend in ("ros", "both"):
            msg = Float32MultiArray()
            msg.data = np.concatenate(
                [
                    self.current_q_or_zeros(),
                    np.zeros(n, dtype=np.float32),
                    np.zeros(n, dtype=np.float32),
                ]
            ).tolist()
            self.joint_cmd_pub.publish(msg)
        if self.motor_backend in ("sdk", "both"):
            self.sdk_motor.set_zero_torque()

    def current_q_or_zeros(self):
        q, _ = self.current_joint_state(fallback_zeros=True)
        return q

    def current_joint_state(self, fallback_zeros=False):
        if self.uses_sdk_joint():
            return (
                np.asarray(self.sdk_motor.dof_pos, dtype=np.float32),
                np.asarray(self.sdk_motor.dof_vel, dtype=np.float32),
            )
        if self.latest_lowstate is not None:
            return (
                self.latest_lowstate[7 : 7 + self.num_dof].astype(np.float32),
                self.latest_lowstate[7 + self.num_dof : 7 + 2 * self.num_dof].astype(np.float32),
            )
        if fallback_zeros:
            return np.zeros(self.num_dof, dtype=np.float32), np.zeros(self.num_dof, dtype=np.float32)
        raise RuntimeError("Joint state is not available from lowstate or SDK.")

    def current_imu_state(self):
        if self.uses_imu_topic():
            imu = self.latest_imu
            gravity = imu[3:6]
            roll = float(np.arctan2(gravity[1], -gravity[2]))
            pitch = float(np.arcsin(np.clip(-gravity[0], -1.0, 1.0)))
            return imu[0:3].astype(np.float32), roll, pitch
        state = self.latest_lowstate
        quat = state[0:4]
        roll, pitch, _ = quat_wxyz_to_roll_pitch_yaw(quat)
        return state[4:7].astype(np.float32), float(roll), float(pitch)

    def current_contacts(self):
        if self.latest_lowstate is None:
            return np.zeros(4, dtype=np.float32)
        return self.latest_lowstate[7 + 2 * self.num_dof : 7 + 2 * self.num_dof + 4].astype(np.float32)

    def uses_sdk_joint(self):
        return self.joint_state_source == "sdk" or (
            self.joint_state_source == "auto" and self.sdk_motor is not None
        )

    def uses_lowstate_joint(self):
        return not self.uses_sdk_joint()

    def uses_imu_topic(self):
        return self.imu_source == "imu" or (
            self.imu_source == "auto" and self.sdk_motor is not None
        )

    def uses_lowstate_imu(self):
        return not self.uses_imu_topic()

    def clip_obs(self, obs):
        return torch.clip(obs, -self.clip_observations, self.clip_observations)


def get_xml_hinge_joint_names_from_path(xml_path):
    import mujoco

    return get_xml_hinge_joint_names(mujoco.MjModel.from_xml_path(xml_path))


def parse_dof_names(value):
    names = [name.strip() for name in value.split(",") if name.strip()]
    if len(names) != 12:
        raise argparse.ArgumentTypeError("--dof_names must contain exactly 12 comma-separated joint names")
    if len(set(names)) != 12:
        raise argparse.ArgumentTypeError("--dof_names contains duplicate joint names")
    return names


def load_dof_names(args):
    if args.dof_names is not None:
        return list(args.dof_names)
    if args.dof_order_source == "default":
        return list(DEFAULT_DOF_NAMES)
    if args.dof_order_source == "sdk_config":
        with open(args.sdk_config, "r") as f:
            sdk_cfg = yaml.safe_load(f)
        names = list(sdk_cfg.get("joint_names", []))
        if len(names) != 12 or len(set(names)) != 12:
            raise ValueError(f"{args.sdk_config} must define 12 unique joint_names")
        return names
    if args.dof_order_source == "xml":
        if not args.mujoco_xml:
            raise ValueError("--mujoco_xml is required when --dof_order_source xml")
        return get_xml_hinge_joint_names_from_path(args.mujoco_xml)
    raise ValueError(f"Unsupported --dof_order_source {args.dof_order_source!r}")


def load_policy(args, n_points, hidden_size):
    model_arg = args.teacher_model if args.policy_type == "teacher" else args.heightmap_model
    model_path = resolve_model_path(args.logdir, model_arg)
    print(f"[model-load] Loading model from {model_path}...", flush=True)
    policy = torch.jit.load(model_path, map_location=args.device)
    policy.eval()
    if getattr(args, "auto_adapt", True):
        obs_dummy = torch.zeros(
            1,
            TEACHER_NUM_PROP + TEACHER_NUM_SCAN + TEACHER_NUM_PRIV_EXPLICIT + TEACHER_NUM_PRIV_LATENT + TEACHER_NUM_HIST * TEACHER_NUM_PROP,
            device=args.device,
        )
        height_dummy = torch.zeros(1, n_points, device=args.device)
        hidden_dummy = torch.zeros(1, 1, hidden_size, device=args.device)
        try:
            with torch.inference_mode():
                if args.policy_type == "teacher":
                    _ = policy(obs_dummy)
                else:
                    _ = policy(obs_dummy, height_dummy, hidden_dummy)
            print(f"[policy-adapt] Loaded model accepted expected signature for policy_type={args.policy_type}", flush=True)
        except Exception:
            if args.policy_type == "teacher":
                def policy_teacher(obs, policy_orig=policy, hd_size=hidden_size):
                    height = obs[:, TEACHER_NUM_PROP : TEACHER_NUM_PROP + TEACHER_NUM_SCAN]
                    hidden = torch.zeros(1, 1, hd_size, device=obs.device)
                    out = policy_orig(obs, height, hidden)
                    return out[0] if isinstance(out, (tuple, list)) else out

                policy = policy_teacher
                print("[policy-adapt] Wrapped student-style model to teacher-style policy(obs)", flush=True)
            else:
                def policy_student(obs, heightmap, hidden, policy_orig=policy):
                    out = policy_orig(obs)
                    return out, hidden

                policy = policy_student
                print("[policy-adapt] Wrapped teacher-style model to student-style policy(obs, heightmap, hidden)", flush=True)
    return policy


@torch.inference_mode()
def main(args):
    with open(osp.join(args.logdir, "config.json"), "r") as f:
        cfg = json.load(f, object_pairs_hook=OrderedDict)
    cfg = apply_task_config_overrides(cfg, args.task_config)
    cfg["control"]["computer_clip_torque"] = True
    n_points = args.n_points
    if n_points is None:
        n_points = int(cfg.get("heightmap", {}).get("n_points", cfg.get("env", {}).get("n_scan", 132)))
    hidden_size = args.hidden_size
    if hidden_size is None:
        hidden_size = int(cfg.get("heightmap_encoder", {}).get("hidden_size", 512))
    policy = load_policy(args, n_points, hidden_size)
    command_controller = KeyboardTeleopController(args) if args.command_source == "keyboard" else None
    if command_controller is None:
        raise ValueError("ROS policy currently supports --command_source keyboard only.")
    sdk_motor = None
    if args.motor_backend in ("sdk", "both"):
        sdk_motor = PythonUnitreeMotorDriver(
            args.sdk_config,
            port0=args.sdk_port0,
            port1=args.sdk_port1,
            baudrate=args.sdk_baudrate,
            timeout=args.sdk_timeout,
        )

    rclpy.init()
    node = RosPolicyNode(args, cfg, policy, n_points, hidden_size, command_controller, sdk_motor=sdk_motor)
    try:
        rclpy.spin(node)
    finally:
        if sdk_motor is not None:
            try:
                sdk_motor.set_zero_torque()
                sdk_motor.set_zero_torque()
            finally:
                sdk_motor.close()
        command_controller.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="traced")
    parser.add_argument("--task_config", type=str, default="mybot_v3", choices=["mybot_v3", "traced"])
    parser.add_argument("--policy_type", type=str, default="student", choices=["student", "teacher"])
    parser.add_argument("--heightmap_model", type=str, default="heightmap_jit.pt")
    parser.add_argument("--teacher_model", type=str, default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--mujoco_xml", type=str, default=None)
    parser.add_argument("--dof_order_source", type=str, default="sdk_config", choices=["sdk_config", "xml", "default"])
    parser.add_argument("--dof_names", type=parse_dof_names, default=None)
    parser.add_argument("--low_state_topic", type=str, default="/lowstate")
    parser.add_argument("--imu_topic", type=str, default="/fast_livo2/state6_imu_prop")
    parser.add_argument("--imu_yaw_correction_deg", type=float, default=0.0)
    parser.add_argument("--joint_cmd_topic", type=str, default="/mujoco/joint_cmd")
    parser.add_argument("--heightmap_topic", type=str, default="/parkour/heightmap_points")
    parser.add_argument("--goal_yaw_topic", type=str, default="/parkour/goal_yaw")
    parser.add_argument("--motor_backend", type=str, default="ros", choices=["ros", "sdk", "both"])
    parser.add_argument("--joint_state_source", type=str, default="auto", choices=["auto", "lowstate", "sdk"])
    parser.add_argument("--imu_source", type=str, default="auto", choices=["auto", "lowstate", "imu"])
    parser.add_argument("--sdk_config", type=str, default=ELMAP_SDK_CONFIG)
    parser.add_argument("--sdk_port0", type=str, default=None)
    parser.add_argument("--sdk_port1", type=str, default=None)
    parser.add_argument("--sdk_baudrate", type=int, default=4000000)
    parser.add_argument("--sdk_timeout", type=float, default=0.02)
    parser.add_argument("--command_source", type=str, default="keyboard", choices=["keyboard"])
    parser.add_argument("--command_vx", type=float, default=0.2)
    parser.add_argument("--command_vy", type=float, default=0.0)
    parser.add_argument("--command_yaw", type=float, default=0.0)
    parser.add_argument("--keyboard_initial_mode", type=int, default=0)
    parser.add_argument("--cmd_vx_min", type=float, default=-0.6)
    parser.add_argument("--cmd_vx_max", type=float, default=0.6)
    parser.add_argument("--cmd_vy_min", type=float, default=-0.6)
    parser.add_argument("--cmd_vy_max", type=float, default=0.6)
    parser.add_argument("--cmd_yaw_min", type=float, default=-1.0)
    parser.add_argument("--cmd_yaw_max", type=float, default=1.0)
    parser.add_argument("--cmd_vx_step", type=float, default=0.1)
    parser.add_argument("--cmd_vy_step", type=float, default=0.1)
    parser.add_argument("--cmd_yaw_step", type=float, default=0.2)
    parser.add_argument("--n_points", type=int, default=None)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--control_dt", type=float, default=0.02)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--auto_adapt", action="store_true", default=True)
    parser.add_argument("--mode", type=str, default="parkour", choices=["parkour", "walk"])
    args = parser.parse_args()
    main(args)
