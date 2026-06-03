import argparse
import json
import os
import os.path as osp
import select
import socket
import struct
import sys
import termios
import tempfile
import threading
import time
import tty
import xml.etree.ElementTree as ET
from collections import OrderedDict

import mujoco
import numpy as np
import torch

from serial_teleop import SerialTeleopController
from unitree_motor_sdk_python import PythonUnitreeMotorDriver


MYBOT_V3_XML = "/home/rc_kfs/extreme-parkour/legged_gym/resources/robots/mybot_v3/xml/mybot_v3.xml"
ELMAP_SDK_CONFIG = "/home/rc_kfs/el_ws/elmap-rl-controller/deploy_cpp/config/robots/mybot_v2_1_cse.yaml"

TEACHER_NUM_PROP = 53
TEACHER_NUM_SCAN = 132
TEACHER_NUM_PRIV_EXPLICIT = 9
TEACHER_NUM_PRIV_LATENT = 29
TEACHER_NUM_HIST = 10
DEFAULT_TEACHER_MODEL = "model_33500_teacher_jit.pt"


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


class FixedTeleopController:
    def __init__(self, mode, vx, vy, yaw, e_stop=False):
        self.cmd = {
            "mode": int(mode),
            "vx": float(vx),
            "vy": float(vy),
            "yaw": float(yaw),
            "e_stop": bool(e_stop),
        }

    def get_latest(self):
        return dict(self.cmd)

    def close(self):
        pass


class UdpTeleopController:
    PACKET = struct.Struct("<ifffB")

    def __init__(self, port, cmd_vx_max, cmd_vy_max, cmd_yaw_max):
        self.cmd_vx_max = float(cmd_vx_max)
        self.cmd_vy_max = float(cmd_vy_max)
        self.cmd_yaw_max = float(cmd_yaw_max)
        self._latest = {"mode": 0, "vx": 0.0, "vy": 0.0, "yaw": 0.0, "e_stop": False}
        self._has_data = False
        self._running = True
        self._lock = threading.Lock()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.settimeout(1.0)
        self._sock.bind(("0.0.0.0", int(port)))
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def has_data(self):
        return self._has_data

    def get_latest(self):
        with self._lock:
            return dict(self._latest)

    def close(self):
        self._running = False
        try:
            self._sock.close()
        except OSError:
            pass
        if self._thread.is_alive():
            self._thread.join(timeout=0.2)

    def _listen_loop(self):
        while self._running:
            try:
                data, _ = self._sock.recvfrom(64)
            except OSError:
                continue
            if len(data) != self.PACKET.size:
                continue
            mode, vx, vy, yaw, e_stop = self.PACKET.unpack(data)
            with self._lock:
                self._latest = {
                    "mode": int(mode),
                    "vx": float(vx) * self.cmd_vx_max,
                    "vy": float(vy) * self.cmd_vy_max,
                    "yaw": float(yaw) * self.cmd_yaw_max,
                    "e_stop": bool(e_stop),
                }
                self._has_data = True


class KeyboardTeleopController:
    def __init__(self, args):
        self.cmd_vx_min = float(args.cmd_vx_min)
        self.cmd_vx_max = float(args.cmd_vx_max)
        self.cmd_vy_min = float(args.cmd_vy_min)
        self.cmd_vy_max = float(args.cmd_vy_max)
        self.cmd_yaw_min = float(args.cmd_yaw_min)
        self.cmd_yaw_max = float(args.cmd_yaw_max)
        self.cmd_vx_step = float(args.cmd_vx_step)
        self.cmd_vy_step = float(args.cmd_vy_step)
        self.cmd_yaw_step = float(args.cmd_yaw_step)
        self._latest = {
            "mode": int(args.keyboard_initial_mode),
            "vx": float(args.command_vx),
            "vy": float(args.command_vy),
            "yaw": float(args.command_yaw),
            "e_stop": False,
        }
        self._running = True
        self._lock = threading.Lock()
        self._fd = sys.stdin.fileno()
        self._old_settings = None
        if sys.stdin.isatty():
            self._old_settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def has_data(self):
        return True

    def get_latest(self):
        with self._lock:
            return dict(self._latest)

    def close(self):
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=0.2)
        if self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
            self._old_settings = None

    def _listen_loop(self):
        while self._running:
            readable, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not readable:
                continue
            key = sys.stdin.read(1)
            if key:
                self._process_key(key)

    def _process_key(self, key):
        with self._lock:
            cmd = self._latest
            if key in ("w", "W"):
                cmd["vx"] = min(cmd["vx"] + self.cmd_vx_step, self.cmd_vx_max)
            elif key in ("s", "S"):
                cmd["vx"] = max(cmd["vx"] - self.cmd_vx_step, self.cmd_vx_min)
            elif key in ("q", "Q"):
                cmd["vy"] = min(cmd["vy"] + self.cmd_vy_step, self.cmd_vy_max)
            elif key in ("e", "E"):
                cmd["vy"] = max(cmd["vy"] - self.cmd_vy_step, self.cmd_vy_min)
            elif key in ("a", "A"):
                cmd["yaw"] = min(cmd["yaw"] + self.cmd_yaw_step, self.cmd_yaw_max)
            elif key in ("d", "D"):
                cmd["yaw"] = max(cmd["yaw"] - self.cmd_yaw_step, self.cmd_yaw_min)
            elif key == "0":
                self._set_mode_locked(0, zero=True)
            elif key == "1":
                self._set_mode_locked(1, zero=True)
            elif key == "2":
                self._set_mode_locked(2, zero=False)
            elif key == "3":
                self._set_mode_locked(3, zero=True)
            elif key in ("r", "R"):
                self._zero_commands_locked()
            elif key == " ":
                self._set_mode_locked(0, zero=True, e_stop=True)
            elif key == "\x1b":
                self._running = False

    def _set_mode_locked(self, mode, zero, e_stop=False):
        self._latest["mode"] = int(mode)
        self._latest["e_stop"] = bool(e_stop)
        if zero:
            self._zero_commands_locked()

    def _zero_commands_locked(self):
        self._latest["vx"] = 0.0
        self._latest["vy"] = 0.0
        self._latest["yaw"] = 0.0


class CommandMux:
    def __init__(self, sources):
        self.sources = sources

    def get_latest(self):
        for source in self.sources:
            if hasattr(source, "has_data") and not source.has_data():
                continue
            return source.get_latest()
        return {"mode": 0, "vx": 0.0, "vy": 0.0, "yaw": 0.0, "e_stop": False}

    def close(self):
        for source in self.sources:
            source.close()


def get_xml_hinge_joint_names(model):
    names = []
    for jid in range(model.njnt):
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name:
            names.append(name)
    return names


def quat_wxyz_to_roll_pitch_yaw(q):
    w, x, y, z = q
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def wrap_to_pi(angle):
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def quat_wxyz_rotate_inverse(q, v):
    w, x, y, z = q
    q_vec = np.array([x, y, z], dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    return (
        v * (2.0 * w * w - 1.0)
        - np.cross(q_vec, v) * (2.0 * w)
        + q_vec * (2.0 * np.dot(q_vec, v))
    )


def parse_debug_items(items):
    if not items:
        return set()
    out = set()
    for item in items.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item == "all":
            return {"cmd", "base", "height", "obs", "action", "target", "joint", "motor", "teacher", "goal"}
        out.add(item)
    return out


class MujocoHeightmapSerialEnv:
    def __init__(
        self,
        xml_path,
        cfg,
        policy,
        device,
        control_dt,
        n_points,
        hidden_size,
        serial_controller,
        height_noise_std=None,
        motor_backend="mujoco",
        sdk_motor=None,
        debug_items=None,
        debug_every=0,
        debug_height_count=12,
        visualize_heightmap=False,
        heightmap_marker_size=0.025,
        visualize_goals=False,
        goal_marker_size=0.08,
        visualize_goal_dirs=False,
        goal_dir_marker_size=0.025,
        external_goals=None,
        input_info_path=None,
        mode="parkour",
        policy_type="student",
        realtime=True,
        render=True,
    ):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.cfg = cfg
        self.policy = policy
        self.device = device
        self.control_dt = control_dt
        self.n_points = n_points
        self.serial_controller = serial_controller
        self.motor_backend = motor_backend
        self.sdk_motor = sdk_motor
        self.debug_items = set(debug_items or [])
        self.debug_every = int(debug_every)
        self.debug_height_count = int(debug_height_count)
        self.visualize_heightmap = bool(visualize_heightmap)
        self.heightmap_marker_size = float(heightmap_marker_size)
        self.visualize_goals = bool(visualize_goals)
        self.goal_marker_size = float(goal_marker_size)
        self.visualize_goal_dirs = bool(visualize_goal_dirs)
        self.goal_dir_marker_size = float(goal_dir_marker_size)
        self.external_goals = None if external_goals is None else np.asarray(external_goals, dtype=np.float32)
        self.input_info_path = input_info_path
        self.input_info_dumped = False
        self.mode = mode
        self.policy_type = policy_type
        self.realtime = realtime
        self.render = render
        self.rng = np.random.default_rng()

        self.reorder_dofs = bool(cfg.get("env", {}).get("reorder_dofs", True))
        self.include_foot_contacts = bool(cfg.get("env", {}).get("include_foot_contacts", True))
        if self.policy_type == "teacher":
            self.include_foot_contacts = True
        self.dof_names = DEPLOY_DOF_NAMES if self.reorder_dofs else get_xml_hinge_joint_names(self.model)
        self.num_dof = len(self.dof_names)
        if self.num_dof != int(cfg.get("env", {}).get("num_actions", 12)):
            raise ValueError(
                f"MuJoCo model has {self.num_dof} hinge joints in policy order, "
                f"but config expects {cfg.get('env', {}).get('num_actions', 12)} actions."
            )
        cfg_n_proprio = int(cfg.get("env", {}).get("n_proprio", 53 if self.include_foot_contacts else 49))
        cfg_hist_len = int(cfg.get("env", {}).get("history_len", 10))
        self.n_proprio = TEACHER_NUM_PROP if self.policy_type == "teacher" else cfg_n_proprio
        self.n_hist_len = TEACHER_NUM_HIST if self.policy_type == "teacher" else cfg_hist_len
        self.n_priv_explicit = 3 + 3 + 3
        self.n_priv_latent = 4 + 1 + 12 + 12
        self.hidden = None if self.policy_type == "teacher" else torch.zeros(1, 1, hidden_size, device=device)
        self.proprio_history_buf = torch.zeros(1, self.n_hist_len, self.n_proprio, device=device)
        self.episode_length_buf = torch.zeros(1, device=device)
        self.actions = torch.zeros(self.num_dof, device=device)
        self.raw_actions = torch.zeros(self.num_dof, device=device)
        self.contact_filt = torch.ones((1, 4), device=device)
        self.last_contacts = np.zeros(4, dtype=bool)
        self.height_points = self._init_height_points()
        heightmap_cfg = cfg.get("heightmap", {})
        self.height_clip_min = float(heightmap_cfg.get("clip_min", -1.0))
        self.height_clip_max = float(heightmap_cfg.get("clip_max", 1.0))
        if height_noise_std is None:
            height_noise_std = float(heightmap_cfg.get("noise_std", 0.0))
        self.height_noise_std = float(height_noise_std)

        self.joint_ids = self._lookup_joint_ids()
        self.qpos_adr = np.array([self.model.jnt_qposadr[jid] for jid in self.joint_ids], dtype=np.int32)
        self.qvel_adr = np.array([self.model.jnt_dofadr[jid] for jid in self.joint_ids], dtype=np.int32)
        self.actuator_ids = self._lookup_actuator_ids()
        self.raycast_bodyexclude = self._find_floating_base_body()
        self.robot_geom_ids = np.nonzero(self.model.geom_bodyid != 0)[0].astype(np.int32)
        self.raycast_geomgroup = np.ones(6, dtype=np.uint8)
        self.raycast_geomgroup[5] = 0
        self.loop_count = 0
        self.last_cmd = {"mode": 0, "vx": 0.0, "vy": 0.0, "yaw": 0.0, "e_stop": False}
        self.last_obs = None
        self.last_proprio = None
        self.last_heightmap = None
        self.last_heightmap_values = None
        self.last_heightmap_world_points = None
        self.last_heightmap_ground_z = None
        self.last_action = None
        self.last_raw_action = None
        self.last_target = None
        self.last_teacher_obs = None
        self.last_priv_explicit = None
        self.last_priv_latent = None
        self.teacher_goals = None
        self.teacher_cur_goal_idx = 0
        self.teacher_reach_goal_timer = 0.0
        self.teacher_delta_yaw = 0.0
        self.teacher_delta_next_yaw = 0.0
        
        # 高度图历史记录
        self.heightmap_history = []  # 记录所有步骤的高度图数据
        self.heightmap_history_max_steps = 10000  # 最多记录 10000 步
        self.last_heightmap_save_time = time.monotonic()

        self.default_dof_pos = torch.tensor(
            [cfg["init_state"]["default_joint_angles"][name] for name in self.dof_names],
            device=device,
            dtype=torch.float32,
        )
        self.p_gains = self._gains_from_config("stiffness")
        self.d_gains = self._gains_from_config("damping")
        self.action_scale = float(cfg["control"]["action_scale"])
        self.clip_actions = float(cfg["normalization"]["clip_actions"])
        self.clip_observations = float(cfg["normalization"].get("clip_observations", 100.0))
        self.obs_scales = cfg["normalization"]["obs_scales"]
        self.torque_limits = self._torque_limits_from_model()

        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        if self.model.nq >= 7:
            init_pos = self.cfg.get("init_state", {}).get("pos", [0.0, 0.0, 0.35])
            self.data.qpos[:3] = np.array(init_pos, dtype=np.float64)
            init_rot_xyzw = self.cfg.get("init_state", {}).get("rot", [0.0, 0.0, 0.0, 1.0])
            self.data.qpos[3:7] = np.array(
                [init_rot_xyzw[3], init_rot_xyzw[0], init_rot_xyzw[1], init_rot_xyzw[2]],
                dtype=np.float64,
            )
        self.data.qpos[self.qpos_adr] = self.default_dof_pos.detach().cpu().numpy()
        self.data.qvel[:] = 0.0
        self.actions.zero_()
        self.raw_actions.zero_()
        self.last_action = None
        self.last_raw_action = None
        mujoco.mj_forward(self.model, self.data)
        self._reset_teacher_goals()

    def step_policy(self):
        cmd = self.serial_controller.get_latest()
        self.last_cmd = dict(cmd)
        if cmd["e_stop"] or int(cmd["mode"]) == 0:
            self.set_zero_torque()
        elif int(cmd["mode"]) == 1 or int(cmd["mode"]) == 4:
            self._apply_pd(self.default_dof_pos, self.p_gains, self.d_gains)
        elif int(cmd["mode"]) == 3:
            self.send_damping(0.2)
        elif int(cmd["mode"]) == 2:
            proprio = self.get_proprio(cmd)
            base_pos, _, yaw = self.get_base_pose()
            heightmap_points = self.sample_heightmap(base_pos, yaw)
            self.last_proprio = proprio.detach().cpu()
            self.last_heightmap = heightmap_points.detach().cpu()
            
            # 将 proprio 转换为 numpy 数组用于索引
            proprio_np = proprio.detach().cpu().numpy()[0] if hasattr(proprio, 'detach') else np.array(proprio)[0]
            
            # 按照 proprioception 布局分解数据
            proprio_breakdown = {
                "base_ang_vel_scaled": proprio_np[0:3].tolist(),
                "roll_pitch": proprio_np[3:5].tolist(),
                "zero_delta_yaw_placeholder": proprio_np[5:6].tolist(),
                "delta_yaw": float(proprio_np[6]),
                "delta_next_yaw": float(proprio_np[7]),
                "zero_command_x_y": proprio_np[8:10].tolist(),
                "command_x": float(proprio_np[10]),
                "parkour_walk_flags": proprio_np[11:13].tolist(),
                "dof_pos_minus_default_scaled": proprio_np[13:25].tolist(),
                "dof_vel_scaled": proprio_np[25:37].tolist(),
                "last_actions": proprio_np[37:49].tolist(),
                "foot_contacts_minus_0p5": proprio_np[49:53].tolist() if self.n_proprio >= 53 else [],
            }
            
            # 记录这一步的完整数据
            step_data = {
                "step": self.loop_count,
                "timestamp": time.monotonic(),
                "base_pos": base_pos.tolist() if isinstance(base_pos, np.ndarray) else list(base_pos),
                "yaw": float(yaw),
                "proprio": proprio_breakdown,
                "heightmap_values": heightmap_points.detach().cpu().numpy().tolist() if hasattr(heightmap_points, 'detach') else heightmap_points.tolist(),
                "command": {
                    "mode": int(cmd["mode"]),
                    "vx": float(cmd["vx"]),
                    "vy": float(cmd["vy"]),
                    "yaw_cmd": float(cmd["yaw"]),
                    "e_stop": bool(cmd["e_stop"]),
                }
            }
            self.heightmap_history.append(step_data)
            
            # 定期保存高度图历史 (每 100 步或 5 秒)
            if self.loop_count % 100 == 0 or time.monotonic() - self.last_heightmap_save_time > 5.0:
                self.save_heightmap_history()
                self.last_heightmap_save_time = time.monotonic()
            
            # 限制内存使用
            if len(self.heightmap_history) > self.heightmap_history_max_steps:
                self.heightmap_history = self.heightmap_history[-self.heightmap_history_max_steps:]
            
            if self.policy_type == "teacher":
                obs = self.build_teacher_obs(proprio, heightmap_points)
                obs = self.clip_obs(obs)
                self.last_obs = obs.detach().cpu()
                self.last_teacher_obs = self.last_obs
                actions = self.policy(obs)
            else:
                obs = self.turn_obs(proprio, self.proprio_history_buf)
                obs = self.clip_obs(obs)
                self.last_obs = obs.detach().cpu()
                actions, self.hidden = self.policy(obs, heightmap_points, self.hidden)
            self.dump_input_info_once(proprio, obs, heightmap_points, actions)
            self.send_action(actions)
        else:
            self._apply_pd(self.default_dof_pos, self.p_gains, self.d_gains)

        substeps = max(1, int(round(self.control_dt / self.model.opt.timestep)))
        for _ in range(substeps):
            mujoco.mj_step(self.model, self.data)
        self.loop_count += 1
        self.debug_print()

    def turn_obs(self, proprio, proprio_history):
        batch_size = proprio.shape[0]
        scan_zeros = torch.zeros(batch_size, self.n_points, device=self.device)
        priv_explicit_zeros = torch.zeros(batch_size, self.n_priv_explicit, device=self.device)
        priv_latent_zeros = torch.zeros(batch_size, self.n_priv_latent, device=self.device)
        return torch.cat(
            [
                proprio,
                scan_zeros,
                priv_explicit_zeros,
                priv_latent_zeros,
                proprio_history.view(batch_size, -1),
            ],
            dim=-1,
        )

    def dump_input_info_once(self, proprio, obs, heightmap_points, actions):
        if self.input_info_dumped or not self.input_info_path:
            return
        
        # Convert heightmap_points to list for serialization
        heightmap_data = heightmap_points.detach().cpu().numpy().tolist() if hasattr(heightmap_points, 'detach') else heightmap_points.tolist()
        
        info = {
            "policy_type": self.policy_type,
            "policy_call": (
                "policy(obs)"
                if self.policy_type == "teacher"
                else "policy(obs, heightmap_points, hidden)"
            ),
            "shapes": {
                "proprio": list(proprio.shape),
                "obs": list(obs.shape),
                "heightmap_points": list(heightmap_points.shape),
                "hidden": None if self.hidden is None else list(self.hidden.shape),
                "actions": list(actions.shape),
                "proprio_history_buf": list(self.proprio_history_buf.shape),
            },
            "configured_dims": {
                "n_proprio": self.n_proprio,
                "n_points": self.n_points,
                "n_priv_explicit": self.n_priv_explicit,
                "n_priv_latent": self.n_priv_latent,
                "history_len": self.n_hist_len,
                "num_dof": self.num_dof,
            },
            "heightmap_points_data": heightmap_data,
            "proprio_layout": [
                {"name": "base_ang_vel_scaled", "range": [0, 3]},
                {"name": "roll_pitch", "range": [3, 5]},
                {"name": "zero_delta_yaw_placeholder", "range": [5, 6]},
                {"name": "delta_yaw", "range": [6, 7]},
                {"name": "delta_next_yaw", "range": [7, 8]},
                {"name": "zero_command_x_y", "range": [8, 10]},
                {"name": "command_x", "range": [10, 11]},
                {"name": "parkour_walk_flags", "range": [11, 13]},
                {"name": "dof_pos_minus_default_scaled", "range": [13, 25]},
                {"name": "dof_vel_scaled", "range": [25, 37]},
                {"name": "last_actions", "range": [37, 49]},
                {"name": "foot_contacts_minus_0p5", "range": [49, 53], "present": self.n_proprio >= 53},
            ],
            "obs_layout": self._obs_layout_for_input_info(),
            "goal": {
                "goal_idx": int(self.teacher_cur_goal_idx),
                "delta_yaw": float(self.teacher_delta_yaw),
                "delta_next_yaw": float(self.teacher_delta_next_yaw),
                "goals_shape": None if self.teacher_goals is None else list(self.teacher_goals.shape),
            },
        }
        path = self.input_info_path
        directory = osp.dirname(osp.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w") as f:
            json.dump(info, f, indent=2)
        self.input_info_dumped = True
        print(
            "[input-info] "
            f"policy_type={self.policy_type} "
            f"proprio={tuple(proprio.shape)} "
            f"obs={tuple(obs.shape)} "
            f"heightmap={tuple(heightmap_points.shape)} "
            f"hidden={None if self.hidden is None else tuple(self.hidden.shape)} "
            f"actions={tuple(actions.shape)} "
            f"saved={path}",
            flush=True,
        )

    def save_heightmap_history(self):
        """保存高度图历史数据到 JSON 文件"""
        if not self.input_info_path or len(self.heightmap_history) == 0:
            return
        
        # 构造输出文件路径 (在 input_info_path 的同级目录)
        base_path = self.input_info_path.replace(".json", "")
        history_path = f"{base_path}_heightmap_history.json"
        
        history_data = {
            "policy_type": self.policy_type,
            "total_steps": self.loop_count,
            "recorded_steps": len(self.heightmap_history),
            "control_dt": float(self.control_dt),
            "heightmap_history": self.heightmap_history,
        }
        
        try:
            directory = osp.dirname(osp.abspath(history_path))
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(history_path, "w") as f:
                json.dump(history_data, f, indent=2)
            print(f"[heightmap-history] Saved {len(self.heightmap_history)} steps to {history_path}", flush=True)
        except Exception as e:
            print(f"[heightmap-history] Error saving history: {e}", flush=True)

    def _obs_layout_for_input_info(self):
        if self.policy_type == "teacher":
            return [
                {"name": "proprio", "range": [0, TEACHER_NUM_PROP]},
                {"name": "scan_heights", "range": [TEACHER_NUM_PROP, TEACHER_NUM_PROP + TEACHER_NUM_SCAN]},
                {
                    "name": "priv_explicit",
                    "range": [
                        TEACHER_NUM_PROP + TEACHER_NUM_SCAN,
                        TEACHER_NUM_PROP + TEACHER_NUM_SCAN + TEACHER_NUM_PRIV_EXPLICIT,
                    ],
                },
                {
                    "name": "priv_latent",
                    "range": [
                        TEACHER_NUM_PROP + TEACHER_NUM_SCAN + TEACHER_NUM_PRIV_EXPLICIT,
                        TEACHER_NUM_PROP
                        + TEACHER_NUM_SCAN
                        + TEACHER_NUM_PRIV_EXPLICIT
                        + TEACHER_NUM_PRIV_LATENT,
                    ],
                },
                {
                    "name": "proprio_history",
                    "range": [
                        TEACHER_NUM_PROP
                        + TEACHER_NUM_SCAN
                        + TEACHER_NUM_PRIV_EXPLICIT
                        + TEACHER_NUM_PRIV_LATENT,
                        TEACHER_NUM_PROP
                        + TEACHER_NUM_SCAN
                        + TEACHER_NUM_PRIV_EXPLICIT
                        + TEACHER_NUM_PRIV_LATENT
                        + TEACHER_NUM_HIST * TEACHER_NUM_PROP,
                    ],
                },
            ]
        return [
            {"name": "proprio", "range": [0, self.n_proprio]},
            {"name": "scan_zeros", "range": [self.n_proprio, self.n_proprio + self.n_points]},
            {
                "name": "priv_explicit_zeros",
                "range": [self.n_proprio + self.n_points, self.n_proprio + self.n_points + self.n_priv_explicit],
            },
            {
                "name": "priv_latent_zeros",
                "range": [
                    self.n_proprio + self.n_points + self.n_priv_explicit,
                    self.n_proprio + self.n_points + self.n_priv_explicit + self.n_priv_latent,
                ],
            },
            {
                "name": "proprio_history",
                "range": [
                    self.n_proprio + self.n_points + self.n_priv_explicit + self.n_priv_latent,
                    self.n_proprio
                    + self.n_points
                    + self.n_priv_explicit
                    + self.n_priv_latent
                    + self.n_hist_len * self.n_proprio,
                ],
            },
        ]

    def build_teacher_obs(self, proprio, heightmap_points):
        batch_size = proprio.shape[0]
        priv_explicit = self.get_priv_explicit()
        priv_latent = self.get_priv_latent()
        obs = torch.cat(
            [
                proprio,
                heightmap_points,
                priv_explicit,
                priv_latent,
                self.proprio_history_buf.view(batch_size, -1),
            ],
            dim=-1,
        )
        expected = (
            TEACHER_NUM_PROP
            + TEACHER_NUM_SCAN
            + TEACHER_NUM_PRIV_EXPLICIT
            + TEACHER_NUM_PRIV_LATENT
            + TEACHER_NUM_HIST * TEACHER_NUM_PROP
        )
        if obs.shape[-1] != expected:
            raise RuntimeError(f"Built teacher obs has {obs.shape[-1]} dims, expected {expected}.")
        return obs

    def get_priv_explicit(self):
        if self.model.nv >= 3 and self.model.nq >= 7:
            base_lin_vel_world = self.data.qvel[:3].copy()
            base_quat = self.data.qpos[3:7].copy()
            base_lin_vel = quat_wxyz_rotate_inverse(base_quat, base_lin_vel_world)
        else:
            base_lin_vel = np.zeros(3, dtype=np.float32)
        lin_vel = torch.tensor(base_lin_vel, device=self.device, dtype=torch.float32).view(1, 3)
        lin_vel = lin_vel * float(self.obs_scales["lin_vel"])
        zeros = torch.zeros_like(lin_vel)
        priv_explicit = torch.cat([lin_vel, zeros, zeros], dim=-1)
        self.last_priv_explicit = priv_explicit.detach().cpu()
        return priv_explicit

    def get_priv_latent(self):
        mass_params = torch.zeros(1, 4, device=self.device)
        friction = torch.tensor([[self._estimate_friction_coeff()]], device=self.device, dtype=torch.float32)
        motor_p = torch.zeros(1, self.num_dof, device=self.device)
        motor_d = torch.zeros(1, self.num_dof, device=self.device)
        priv_latent = torch.cat([mass_params, friction, motor_p, motor_d], dim=-1)
        self.last_priv_latent = priv_latent.detach().cpu()
        return priv_latent

    def _estimate_friction_coeff(self):
        world_geom_ids = np.nonzero(self.model.geom_bodyid == 0)[0]
        if world_geom_ids.size:
            values = self.model.geom_friction[world_geom_ids, 0]
            if values.size:
                return float(np.mean(values))
        if self.model.ngeom:
            return float(np.mean(self.model.geom_friction[:, 0]))
        return 1.0

    def _reset_teacher_goals(self):
        base_pos, _, _ = self.get_base_pose()
        if self.external_goals is not None:
            goals = self.external_goals.copy()
            goals[:, 0] += base_pos[0]
            goals[:, 1] += base_pos[1]
            self.teacher_goals = goals
            self.teacher_cur_goal_idx = 0
            self.teacher_reach_goal_timer = 0.0
            self.teacher_delta_yaw = 0.0
            self.teacher_delta_next_yaw = 0.0
            return
        terrain_cfg = self.cfg.get("terrain", {})
        num_goals = int(terrain_cfg.get("num_goals", 8))
        terrain_length = float(terrain_cfg.get("terrain_length", 18.0))
        x_goals = np.linspace(1.0, max(1.0, terrain_length - 1.0), num_goals, dtype=np.float32)
        goals = np.zeros((num_goals, 3), dtype=np.float32)
        goals[:, 0] = base_pos[0] + x_goals
        goals[:, 1] = base_pos[1]
        self.teacher_goals = goals
        self.teacher_cur_goal_idx = 0
        self.teacher_reach_goal_timer = 0.0
        self.teacher_delta_yaw = 0.0
        self.teacher_delta_next_yaw = 0.0

    def _update_teacher_goals(self):
        if self.teacher_goals is None:
            self._reset_teacher_goals()
        base_pos, _, yaw = self.get_base_pose()
        env_cfg = self.cfg.get("env", {})
        threshold = float(env_cfg.get("next_goal_threshold", 0.2))
        reach_delay = float(env_cfg.get("reach_goal_delay", 0.1))
        goal_idx = min(self.teacher_cur_goal_idx, len(self.teacher_goals) - 1)
        cur_goal = self.teacher_goals[goal_idx]
        if np.linalg.norm(base_pos[:2] - cur_goal[:2]) < threshold:
            self.teacher_reach_goal_timer += self.control_dt
            if self.teacher_reach_goal_timer > reach_delay and self.teacher_cur_goal_idx < len(self.teacher_goals) - 1:
                self.teacher_cur_goal_idx += 1
                self.teacher_reach_goal_timer = 0.0
        else:
            self.teacher_reach_goal_timer = 0.0

        goal_idx = min(self.teacher_cur_goal_idx, len(self.teacher_goals) - 1)
        next_idx = min(goal_idx + 1, len(self.teacher_goals) - 1)
        cur_goal = self.teacher_goals[goal_idx]
        next_goal = self.teacher_goals[next_idx]
        target_vec = cur_goal[:2] - base_pos[:2]
        next_vec = next_goal[:2] - base_pos[:2]
        target_yaw = np.arctan2(target_vec[1], target_vec[0])
        next_target_yaw = np.arctan2(next_vec[1], next_vec[0])
        self.teacher_delta_yaw = wrap_to_pi(target_yaw - yaw)
        self.teacher_delta_next_yaw = wrap_to_pi(next_target_yaw - yaw)

    def get_proprio(self, cmd):
        if self.model.nv >= 6:
            ang_vel_np = self.data.qvel[3:6].copy()
        else:
            ang_vel_np = np.zeros(3, dtype=np.float32)
        ang_vel = torch.tensor(ang_vel_np, device=self.device, dtype=torch.float32).view(1, 3)
        ang_vel = ang_vel * float(self.obs_scales["ang_vel"])

        if self.model.nq >= 7:
            roll, pitch, _ = quat_wxyz_to_roll_pitch_yaw(self.data.qpos[3:7])
        else:
            roll, pitch = 0.0, 0.0
        imu = torch.tensor([[roll, pitch]], device=self.device, dtype=torch.float32)
        self._update_teacher_goals()
        yaw_info = torch.tensor(
            [[0.0, self.teacher_delta_yaw, self.teacher_delta_next_yaw]],
            device=self.device,
            dtype=torch.float32,
        )
        commands = torch.tensor([[0.0, 0.0, float(cmd["vx"])]], device=self.device)
        parkour_walk = torch.tensor(
            [[1.0, 0.0] if self.mode == "parkour" else [0.0, 1.0]],
            device=self.device,
            dtype=torch.float32,
        )
        dof_pos = (self.current_dof_pos().view(1, -1) - self.default_dof_pos.view(1, -1))
        dof_pos = dof_pos * float(self.obs_scales["dof_pos"])
        dof_vel = self.current_dof_vel().view(1, -1) * float(self.obs_scales["dof_vel"])
        last_actions = self.raw_actions.view(1, -1)
        proprio_parts = [ang_vel, imu, yaw_info, commands, parkour_walk, dof_pos, dof_vel, last_actions]
        if self.n_proprio >= 53:
            proprio_parts.append(self._contact_obs())

        proprio = torch.cat(proprio_parts, dim=-1)
        if proprio.shape[-1] != self.n_proprio:
            raise RuntimeError(f"Built proprio has {proprio.shape[-1]} dims, config expects {self.n_proprio}.")
        self.proprio_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([proprio] * self.n_hist_len, dim=1),
            torch.cat([self.proprio_history_buf[:, 1:], proprio.unsqueeze(1)], dim=1),
        )
        self.episode_length_buf += 1
        return proprio

    def send_action(self, actions):
        hard_clip = self.clip_actions / self.action_scale
        raw_actions = actions.view(-1).to(self.device)
        clipped_actions = torch.clip(raw_actions, -hard_clip, hard_clip)
        self.raw_actions = raw_actions
        self.actions = clipped_actions
        self.last_raw_action = raw_actions.detach().cpu()
        self.last_action = clipped_actions.detach().cpu()
        target = clipped_actions * self.action_scale + self.default_dof_pos
        self._apply_pd(target, self.p_gains, self.d_gains)

    def clip_obs(self, obs):
        return torch.clip(obs, -self.clip_observations, self.clip_observations)

    def current_dof_pos(self):
        if self.motor_backend == "sdk" and self.sdk_motor is not None:
            return torch.tensor(self.sdk_motor.dof_pos, device=self.device, dtype=torch.float32)
        return torch.tensor(self.data.qpos[self.qpos_adr].copy(), device=self.device, dtype=torch.float32)

    def current_dof_vel(self):
        if self.motor_backend == "sdk" and self.sdk_motor is not None:
            return torch.tensor(self.sdk_motor.dof_vel, device=self.device, dtype=torch.float32)
        return torch.tensor(self.data.qvel[self.qvel_adr].copy(), device=self.device, dtype=torch.float32)

    def get_base_pose(self):
        if self.model.nq < 7:
            return np.zeros(3, dtype=np.float32), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), 0.0
        base_pos = self.data.qpos[:3].copy().astype(np.float32)
        quat = self.data.qpos[3:7].copy().astype(np.float32)
        _, _, yaw = quat_wxyz_to_roll_pitch_yaw(quat)
        return base_pos, quat, float(yaw)

    def sample_heightmap(self, base_pos, yaw):
        """Sample the same scandot layout as the IsaacGym heightmap student.

        IsaacGym computes:
          height = clip(base_z - 0.3 - measured_terrain_height, -1, 1)
        Temporarily use base_z - measured_terrain_height for easier MuJoCo
        debug; restore the 0.3 offset to match training.

        The local measured_points_x/y grid is rotated by base yaw, translated
        into world XY, then raycast downward in MuJoCo to find terrain height.
        """
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

        heights = np.clip(
            # Training formula: float(base_pos[2]) - 0.3 - measured
            float(base_pos[2]) - 0.3 - measured,
            self.height_clip_min,
            self.height_clip_max,
        ).astype(np.float32)
        if self.height_noise_std > 0.0:
            heights = heights + self.rng.normal(0.0, self.height_noise_std, size=heights.shape).astype(np.float32)
            heights = np.clip(heights, self.height_clip_min, self.height_clip_max).astype(np.float32)
        self.last_heightmap_values = heights.copy()
        self.last_heightmap_ground_z = measured.copy()
        self.last_heightmap_world_points = np.column_stack([world_xy, measured]).astype(np.float32)
        return torch.from_numpy(heights).to(self.device).view(1, -1)

    def _apply_pd(self, target, kp, kd):
        target = target.to(self.device)
        self.last_target = target.detach().cpu()
        if self.motor_backend in ("sdk", "both") and self.sdk_motor is not None:
            self.sdk_motor.send_commands(
                target.detach().cpu().tolist(),
                kp.detach().cpu().tolist(),
                kd.detach().cpu().tolist(),
            )
        if self.motor_backend in ("mujoco", "both"):
            q = self.current_dof_pos()
            dq = self.current_dof_vel()
            tau = kp * (target - q) - kd * dq
            tau = torch.clip(tau, -self.torque_limits, self.torque_limits)
            tau_np = tau.detach().cpu().numpy()
            for dof_idx, actuator_id in enumerate(self.actuator_ids):
                self.data.ctrl[actuator_id] = tau_np[dof_idx]

    def send_damping(self, kd):
        if self.motor_backend in ("sdk", "both") and self.sdk_motor is not None:
            self.sdk_motor.send_damping(kd)
        if self.motor_backend in ("mujoco", "both"):
            dq = torch.tensor(self.data.qvel[self.qvel_adr].copy(), device=self.device, dtype=torch.float32)
            tau = -torch.ones_like(self.d_gains) * float(kd) * dq
            tau_np = tau.detach().cpu().numpy()
            for dof_idx, actuator_id in enumerate(self.actuator_ids):
                self.data.ctrl[actuator_id] = tau_np[dof_idx]

    def set_zero_torque(self):
        if self.motor_backend in ("sdk", "both") and self.sdk_motor is not None:
            self.sdk_motor.set_zero_torque()
        if self.motor_backend in ("mujoco", "both"):
            self.data.ctrl[:] = 0.0

    def debug_print(self):
        if self.debug_every <= 0 or not self.debug_items:
            return
        if self.loop_count % self.debug_every != 0:
            return
        base_pos, _, yaw = self.get_base_pose()
        parts = [f"[debug step={self.loop_count} mode={self.last_cmd.get('mode')} backend={self.motor_backend}]"]
        if "cmd" in self.debug_items:
            parts.append(
                "cmd="
                f"vx:{self.last_cmd.get('vx', 0.0):.3f} "
                f"vy:{self.last_cmd.get('vy', 0.0):.3f} "
                f"yaw:{self.last_cmd.get('yaw', 0.0):.3f} "
                f"estop:{int(bool(self.last_cmd.get('e_stop', False)))}"
            )
        if "base" in self.debug_items:
            parts.append(f"base=pos[{base_pos[0]:.3f},{base_pos[1]:.3f},{base_pos[2]:.3f}] yaw:{yaw:.3f}")
        if "height" in self.debug_items and self.last_heightmap_values is not None:
            h = self.last_heightmap_values
            n = min(self.debug_height_count, h.shape[0])
            vals = np.array2string(h[:n], precision=3, separator=",")
            parts.append(f"height=min:{h.min():.3f} max:{h.max():.3f} mean:{h.mean():.3f} first{n}:{vals}")
        if "obs" in self.debug_items and self.last_obs is not None:
            obs = self.last_obs.view(-1)
            parts.append(f"obs=shape:{tuple(self.last_obs.shape)} min:{obs.min():.3f} max:{obs.max():.3f} mean:{obs.mean():.3f}")
        if "teacher" in self.debug_items and self.policy_type == "teacher" and self.last_obs is not None:
            obs = self.last_obs.view(-1)
            scan_start = TEACHER_NUM_PROP
            scan_end = scan_start + TEACHER_NUM_SCAN
            priv_start = scan_end
            priv_end = priv_start + TEACHER_NUM_PRIV_EXPLICIT + TEACHER_NUM_PRIV_LATENT
            parts.append(
                "teacher="
                f"obs:{tuple(self.last_obs.shape)} "
                f"prop:{obs[:TEACHER_NUM_PROP].abs().max():.3f} "
                f"scan_minmax:[{obs[scan_start:scan_end].min():.3f},{obs[scan_start:scan_end].max():.3f}] "
                f"priv_absmax:{obs[priv_start:priv_end].abs().max():.3f} "
                f"goal_idx:{self.teacher_cur_goal_idx} "
                f"dyaw:{self.teacher_delta_yaw:.3f} "
                f"dnext:{self.teacher_delta_next_yaw:.3f}"
            )
        if "goal" in self.debug_items and self.teacher_goals is not None:
            goal_idx = min(self.teacher_cur_goal_idx, len(self.teacher_goals) - 1)
            next_idx = min(goal_idx + 1, len(self.teacher_goals) - 1)
            cur = self.teacher_goals[goal_idx]
            nxt = self.teacher_goals[next_idx]
            parts.append(
                "goal="
                f"idx:{goal_idx} "
                f"cur:[{cur[0]:.2f},{cur[1]:.2f},{cur[2]:.2f}] "
                f"next:[{nxt[0]:.2f},{nxt[1]:.2f},{nxt[2]:.2f}] "
                f"dyaw:{self.teacher_delta_yaw:.3f} "
                f"dnext:{self.teacher_delta_next_yaw:.3f}"
            )
        if "action" in self.debug_items and self.last_action is not None:
            a = self.last_action.view(-1)
            msg = f"action=clip_abs:{a.abs().max():.3f} first3:{a[:3].numpy()}"
            if self.last_raw_action is not None:
                raw = self.last_raw_action.view(-1)
                msg += f" raw_abs:{raw.abs().max():.3f} raw_first3:{raw[:3].numpy()}"
            parts.append(msg)
        if "target" in self.debug_items and self.last_target is not None:
            t = self.last_target.view(-1)
            parts.append(f"target=first3:{t[:3].numpy()}")
        if "joint" in self.debug_items:
            q = self.current_dof_pos().detach().cpu()
            dq = self.current_dof_vel().detach().cpu()
            parts.append(f"joint=q0-2:{q[:3].numpy()} dq0-2:{dq[:3].numpy()}")
        if "motor" in self.debug_items and self.sdk_motor is not None:
            parts.append(
                "motor="
                f"temp0-2:{self.sdk_motor.motor_temps[:3]} "
                f"err:{self.sdk_motor.motor_errors}"
            )
        print(" | ".join(parts), flush=True)

    def update_heightmap_markers(self, viewer):
        if not self.visualize_heightmap and not self.visualize_goals:
            return
        if self.last_heightmap_world_points is None and self.teacher_goals is None:
            return
        scene = viewer.user_scn
        scene.ngeom = 0
        mat = np.eye(3, dtype=np.float64).reshape(-1)
        if self.visualize_heightmap and self.last_heightmap_world_points is not None:
            size = np.array([self.heightmap_marker_size] * 3, dtype=np.float64)
            values = self.last_heightmap_values
            points = self.last_heightmap_world_points
            for idx, point in enumerate(points):
                if scene.ngeom >= scene.maxgeom:
                    break
                value = 0.0 if values is None else float(values[idx])
                normalized = (value - self.height_clip_min) / max(self.height_clip_max - self.height_clip_min, 1e-6)
                normalized = float(np.clip(normalized, 0.0, 1.0))
                rgba = np.array([normalized, 0.2, 1.0 - normalized, 1.0], dtype=np.float32)
                pos = np.array([point[0], point[1], point[2] + self.heightmap_marker_size], dtype=np.float64)
                mujoco.mjv_initGeom(
                    scene.geoms[scene.ngeom],
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    size,
                    pos,
                    mat,
                    rgba,
                )
                scene.ngeom += 1
        if self.visualize_goals and self.teacher_goals is not None:
            self._add_goal_markers(scene, mat)
        if self.visualize_goal_dirs and self.teacher_goals is not None:
            self._add_goal_direction_markers(scene, mat)

    def _add_goal_markers(self, scene, mat):
        size = np.array([self.goal_marker_size] * 3, dtype=np.float64)
        goal_idx = min(self.teacher_cur_goal_idx, len(self.teacher_goals) - 1)
        next_idx = min(goal_idx + 1, len(self.teacher_goals) - 1)
        for idx, goal in enumerate(self.teacher_goals):
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
            pos = np.array([goal[0], goal[1], goal[2] + self.goal_marker_size], dtype=np.float64)
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                size * (1.35 if idx == goal_idx else 1.0),
                pos,
                mat,
                rgba,
            )
            scene.ngeom += 1

    def _add_goal_direction_markers(self, scene, mat):
        base_pos, _, _ = self.get_base_pose()
        goal_idx = min(self.teacher_cur_goal_idx, len(self.teacher_goals) - 1)
        next_idx = min(goal_idx + 1, len(self.teacher_goals) - 1)
        cur_vec = self.teacher_goals[goal_idx, :2] - base_pos[:2]
        next_vec = self.teacher_goals[next_idx, :2] - base_pos[:2]
        cur_norm = np.linalg.norm(cur_vec)
        next_norm = np.linalg.norm(next_vec)
        if cur_norm > 1e-5:
            cur_dir = cur_vec / cur_norm
            self._add_goal_dir_dots(
                scene,
                mat,
                base_pos,
                cur_dir,
                spacing=0.1,
                start_idx=3,
                rgba=np.array([1.0, 0.35, 0.25, 1.0], dtype=np.float32),
            )
        if next_norm > 1e-5:
            next_dir = next_vec / next_norm
            self._add_goal_dir_dots(
                scene,
                mat,
                base_pos,
                next_dir,
                spacing=0.2,
                start_idx=3,
                rgba=np.array([0.0, 1.0, 0.5, 1.0], dtype=np.float32),
            )

    def _add_goal_dir_dots(self, scene, mat, base_pos, direction, spacing, start_idx, rgba):
        size = np.array([self.goal_dir_marker_size] * 3, dtype=np.float64)
        for i in range(5):
            if scene.ngeom >= scene.maxgeom:
                break
            xy = base_pos[:2] + spacing * (i + start_idx) * direction
            pos = np.array([xy[0], xy[1], base_pos[2]], dtype=np.float64)
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                size,
                pos,
                mat,
                rgba,
            )
            scene.ngeom += 1

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
        if points.shape[0] != self.n_points:
            raise ValueError(f"Heightmap grid has {points.shape[0]} points, expected {self.n_points}.")
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

    def _torque_limits_from_model(self):
        limits = np.full(self.num_dof, np.inf, dtype=np.float32)
        for dof_idx, actuator_id in enumerate(self.actuator_ids):
            if self.model.actuator_forcelimited[actuator_id]:
                limits[dof_idx] = float(self.model.actuator_forcerange[actuator_id, 1])
        if np.isinf(limits).any():
            joint_limits = self.cfg.get("control", {}).get("torque_limits", None)
            if joint_limits is not None:
                limits[np.isinf(limits)] = float(joint_limits)
            else:
                limits[np.isinf(limits)] = 33.5
        return torch.tensor(limits, device=self.device, dtype=torch.float32)

    def _contact_obs(self):
        if not self.include_foot_contacts:
            return torch.full((1, 4), -0.5, device=self.device, dtype=torch.float32)
        feet = ["FR", "FL", "RR", "RL"]
        contact_bool = np.zeros(4, dtype=bool)
        for i in range(self.data.ncon):
            mj_contact = self.data.contact[i]
            geom1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, mj_contact.geom1) or ""
            geom2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, mj_contact.geom2) or ""
            pair_name = f"{geom1} {geom2}"
            for foot_idx, prefix in enumerate(feet):
                if prefix in pair_name:
                    contact_bool[foot_idx] = True
        filtered = np.logical_or(contact_bool, self.last_contacts)
        self.last_contacts = contact_bool
        contact = filtered.astype(np.float32) - 0.5
        return torch.tensor(contact, device=self.device, dtype=torch.float32).view(1, 4)


def maybe_import_viewer(render):
    if not render:
        return None
    import mujoco.viewer

    return mujoco.viewer


def resolve_model_path(logdir, model_path):
    if osp.isabs(model_path):
        return model_path
    return osp.join(logdir, model_path)


def generate_parkour_course(cfg, seed=None, difficulty=0.5):
    rng = np.random.default_rng(seed)
    terrain_cfg = cfg.get("terrain", {})
    num_goals = int(terrain_cfg.get("num_goals", 8))
    num_stones = max(1, num_goals - 2)
    horizontal_scale = float(terrain_cfg.get("horizontal_scale", 0.05))
    if difficulty < 0.0:
        # play.py sets max_difficulty=True, which samples difficulty in [0.7, 1.0].
        difficulty = float(rng.uniform(0.7, 1.0))

    # Matches Terrain.make_terrain() for terrain_dict {"parkour": 1.0}.
    x_range = [-0.1, 0.1 + 0.3 * difficulty]
    y_range = [0.2, 0.3 + 0.1 * difficulty]
    stone_len_range = [0.9 - 0.3 * difficulty, 1.0 - 0.2 * difficulty]
    incline_height = 0.25 * difficulty
    last_incline_height = incline_height + 0.1 - 0.1 * difficulty
    stone_width = 1.0
    platform_len = 2.5
    platform_height = 0.0
    last_stone_len = 1.6
    pit_depth = float(rng.uniform(0.2, 1.0))
    robot_origin_x = 1.0

    stone_len = float(rng.uniform(*stone_len_range))
    stone_len = 2.0 * round(stone_len / 2.0, 1)
    dis_x_min = stone_len + x_range[0]
    dis_x_max = stone_len + x_range[1]
    dis_y_min, dis_y_max = y_range

    goals = np.zeros((num_stones + 2, 3), dtype=np.float32)
    geoms = []

    geoms.append(
        {
            "name": "parkour_start_platform",
            "pos": [platform_len / 2.0 - robot_origin_x, 0.0, platform_height / 2.0],
            "size": [platform_len / 2.0, 2.0, 0.025],
            "rgba": "0.45 0.45 0.45 1",
        }
    )
    goals[0] = [platform_len - stone_len / 2.0 - robot_origin_x, 0.0, platform_height]

    dis_x = platform_len - float(rng.uniform(dis_x_min, dis_x_max)) + stone_len / 2.0
    left_right_flag = int(rng.integers(0, 2))
    dis_z = 0.0
    last_center_x = dis_x
    last_len = stone_len
    for i in range(num_stones):
        dis_x += float(rng.uniform(dis_x_min, dis_x_max))
        pos_neg = 1.0 if left_right_flag == 1 else -1.0
        dis_y = pos_neg * float(rng.uniform(dis_y_min, dis_y_max))
        if i == num_stones - 1:
            dis_x += last_stone_len / 4.0
            length = last_stone_len
            height = last_incline_height
        else:
            length = stone_len
            height = incline_height
        slope_angle = float(np.arctan2(2.0 * height, stone_width) * pos_neg)
        goals[i + 1] = [dis_x - robot_origin_x, dis_y, dis_z]
        geoms.append(
            {
                "name": f"parkour_stone_{i}",
                "pos": [dis_x - robot_origin_x, dis_y, dis_z],
                "size": [length / 2.0, stone_width / 2.0, 0.03],
                "euler": [slope_angle, 0.0, 0.0],
                "rgba": "0.35 0.35 0.35 1",
            }
        )
        last_center_x = dis_x
        last_len = length
        left_right_flag = 1 - left_right_flag

    final_dis_x = last_center_x + 2.0 * float(rng.uniform(dis_x_min, dis_x_max))
    final_platform_start = last_center_x + last_len / 2.0 + 0.05
    final_len = max(3.0, float(terrain_cfg.get("terrain_length", 18.0)) - final_platform_start)
    geoms.append(
        {
            "name": "parkour_final_platform",
            "pos": [final_platform_start + final_len / 2.0 - robot_origin_x, 0.0, platform_height / 2.0],
            "size": [final_len / 2.0, 2.0, 0.025],
            "rgba": "0.45 0.45 0.45 1",
        }
    )
    goals[-1] = [final_dis_x - robot_origin_x, 0.0, platform_height]
    return {"geoms": geoms, "goals": goals, "pit_depth": pit_depth, "horizontal_scale": horizontal_scale}


def build_mujoco_xml_with_terrain(xml_path, terrain, normalize_dynamics=True):
    if terrain is None and not normalize_dynamics:
        return xml_path
    tree = ET.parse(xml_path)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is not None:
        meshdir = compiler.get("meshdir")
        if meshdir and not osp.isabs(meshdir):
            compiler.set("meshdir", osp.abspath(osp.join(osp.dirname(xml_path), meshdir)))

    if normalize_dynamics:
        for joint in root.findall(".//joint"):
            if joint.get("type") == "free":
                continue
            joint.set("damping", "0")
            joint.set("frictionloss", "0")
            joint.set("armature", "0")
        for motor in root.findall(".//motor"):
            motor.set("gear", "1")
            motor.set("ctrlrange", "-33.5 33.5")
            motor.set("ctrllimited", "true")

    if terrain is not None:
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError("MuJoCo XML has no worldbody.")
        floor = worldbody.find("./geom[@name='floor']")
        if floor is not None:
            floor.set("pos", f"0 0 {-terrain['pit_depth']:.6f}")
            floor.set("rgba", "0.08 0.08 0.08 1")
        for geom_cfg in terrain["geoms"]:
            ET.SubElement(
                worldbody,
                "geom",
                {
                    "name": geom_cfg["name"],
                    "type": "box",
                    "pos": "{:.6f} {:.6f} {:.6f}".format(*geom_cfg["pos"]),
                    "size": "{:.6f} {:.6f} {:.6f}".format(*geom_cfg["size"]),
                    "euler": "{:.6f} {:.6f} {:.6f}".format(*geom_cfg.get("euler", [0.0, 0.0, 0.0])),
                    "rgba": geom_cfg["rgba"],
                    "contype": "1",
                    "conaffinity": "1",
                    "condim": "3",
                    "friction": "1.0 0.3 0.3",
                },
            )

    tmp = tempfile.NamedTemporaryFile(prefix="mujoco_runtime_", suffix=".xml", dir="/tmp", delete=False)
    tmp.close()
    tree.write(tmp.name, encoding="utf-8", xml_declaration=True)
    return tmp.name


def apply_task_config_overrides(cfg, task_config):
    if task_config == "traced":
        return cfg
    if task_config != "mybot_v3":
        raise ValueError(f"Unsupported --task_config {task_config!r}.")

    cfg.setdefault("env", {})
    cfg["env"]["reorder_dofs"] = False
    cfg["env"]["include_foot_contacts"] = False
    cfg["env"]["n_proprio"] = 53
    cfg["env"]["num_observations"] = 753

    cfg.setdefault("heightmap", {})
    cfg["heightmap"].update(
        {
            "use_heightmap": True,
            "n_points": 132,
            "update_interval": 5,
            "buffer_len": 3,
            "noise_std": 0.02,
            "horizontal_noise": 0.02,
            "clip_min": -1.0,
            "clip_max": 1.0,
        }
    )

    cfg.setdefault("init_state", {})
    cfg["init_state"]["pos"] = [0.0, 0.0, 0.45]
    cfg["init_state"]["default_joint_angles"] = OrderedDict(
        [
            ("FL_hip_joint", 0.1),
            ("RL_hip_joint", 0.1),
            ("FR_hip_joint", -0.1),
            ("RR_hip_joint", -0.1),
            ("FL_thigh_joint", 0.8),
            ("RL_thigh_joint", 1.0),
            ("FR_thigh_joint", 0.8),
            ("RR_thigh_joint", 1.0),
            ("FL_calf_joint", -1.5),
            ("RL_calf_joint", -1.5),
            ("FR_calf_joint", -1.5),
            ("RR_calf_joint", -1.5),
        ]
    )

    cfg.setdefault("control", {})
    cfg["control"].update(
        {
            "control_type": "P",
            "stiffness": {"joint": 35.0},
            "damping": {"joint": 1.0},
            "action_scale": 0.25,
            "decimation": 4,
        }
    )

    cfg.setdefault("asset", {})
    cfg["asset"].update(
        {
            "file": "{LEGGED_GYM_ROOT_DIR}/resources/robots/mybot_v3/urdf/mybot_v3.urdf",
            "foot_name": "foot",
            "terminate_after_contacts_on": ["body"],
            "collapse_fixed_joints": False,
            "flip_visual_attachments": False,
        }
    )
    return cfg


@torch.inference_mode()
def main(args):
    with open(osp.join(args.logdir, "config.json"), "r") as f:
        cfg = json.load(f, object_pairs_hook=OrderedDict)
    cfg = apply_task_config_overrides(cfg, args.task_config)
    cfg["control"]["computer_clip_torque"] = True
    n_points = args.n_points
    if n_points is None:
        n_points = int(cfg.get("heightmap", {}).get("n_points", cfg.get("env", {}).get("n_scan", 132)))
    if args.policy_type == "teacher" and n_points != TEACHER_NUM_SCAN:
        raise ValueError(f"Teacher JIT expects {TEACHER_NUM_SCAN} scan points, got {n_points}.")
    hidden_size = args.hidden_size
    if hidden_size is None:
        hidden_size = int(cfg.get("heightmap_encoder", {}).get("hidden_size", 512))

    device = args.device
    model_arg = args.teacher_model if args.policy_type == "teacher" else args.heightmap_model
    model_path = resolve_model_path(args.logdir, model_arg)
    print(f"[model-load] Loading model from {model_path}...", flush=True)
    policy = torch.jit.load(model_path, map_location=device)
    policy.eval()
    print(f"[model-load] Model loaded successfully", flush=True)
    
    # Optimize RNN weights memory layout
    try:
        if hasattr(policy, 'flatten_parameters'):
            policy.flatten_parameters()
            print(f"[model-load] Flattened RNN parameters", flush=True)
    except Exception as e:
        print(f"[model-load] Note: Could not flatten parameters: {e}", flush=True)

    # Auto-adapt loaded policy to the expected calling convention.
    # If args.auto_adapt is True, try calling the model with the expected
    # signature and wrap it if the signatures don't match.
    if getattr(args, "auto_adapt", True):
        obs_dummy = torch.zeros(
            1,
            TEACHER_NUM_PROP + TEACHER_NUM_SCAN + TEACHER_NUM_PRIV_EXPLICIT + TEACHER_NUM_PRIV_LATENT + TEACHER_NUM_HIST * TEACHER_NUM_PROP,
            device=device,
        )
        height_dummy = torch.zeros(1, n_points, device=device)
        hidden_dummy = torch.zeros(1, 1, hidden_size, device=device)
        adapted = False
        try:
            if args.policy_type == "teacher":
                with torch.inference_mode():
                    _ = policy(obs_dummy)
            else:
                with torch.inference_mode():
                    _ = policy(obs_dummy, height_dummy, hidden_dummy)
            adapted = True
            print(f"[policy-adapt] Loaded model at {model_path} accepted expected signature for policy_type={args.policy_type}")
        except Exception:
            # Need to wrap model to expected signature
            if args.policy_type == "teacher":
                # env expects teacher(obs) but model is likely student(obs, height, hidden)
                def policy_teacher(obs, policy_orig=policy, hd_size=hidden_size):
                    height = obs[:, TEACHER_NUM_PROP : TEACHER_NUM_PROP + TEACHER_NUM_SCAN]
                    hidden = torch.zeros(1, 1, hd_size, device=obs.device)
                    out = policy_orig(obs, height, hidden)
                    if isinstance(out, (tuple, list)):
                        return out[0]
                    return out

                policy = policy_teacher
                adapted = True
                print("[policy-adapt] Wrapped student-style model to teacher-style policy(obs)")
            else:
                # env expects student(obs, height, hidden) but model is likely teacher(obs)
                def policy_student(obs, heightmap, hidden, policy_orig=policy):
                    out = policy_orig(obs)
                    return out, hidden

                policy = policy_student
                adapted = True
                print("[policy-adapt] Wrapped teacher-style model to student-style policy(obs, heightmap, hidden)")

        if adapted:
            # quick verification
            try:
                with torch.inference_mode():
                    if args.policy_type == "teacher":
                        _ = policy(obs_dummy)
                    else:
                        _ = policy(obs_dummy, height_dummy, hidden_dummy)
                print("[policy-adapt] Verification forward succeeded.")
            except Exception as e:
                print(f"[policy-adapt] Verification forward FAILED: {e}")

    print("[init] Model ready. Starting environment initialization...", flush=True)
    
    terrain = None
    mujoco_xml = args.mujoco_xml
    if args.mujoco_terrain == "parkour":
        print(f"[init] Generating parkour terrain (difficulty={args.terrain_difficulty}, seed={args.terrain_seed})...", flush=True)
        terrain = generate_parkour_course(cfg, seed=args.terrain_seed, difficulty=args.terrain_difficulty)
        print("[init] Terrain ready", flush=True)
    else:
        print("[init] Using flat terrain", flush=True)
    if args.normalize_mujoco_xml or terrain is not None:
        print("[init] Building runtime MuJoCo XML...", flush=True)
        mujoco_xml = build_mujoco_xml_with_terrain(args.mujoco_xml, terrain, normalize_dynamics=args.normalize_mujoco_xml)
        
    input_info_path = args.input_info_path
    if input_info_path == "auto":
        input_info_path = osp.join(args.logdir, f"mujoco_input_dims_{args.policy_type}.json")

    sdk_motor = None
    if args.motor_backend in ("sdk", "both"):
        sdk_motor = PythonUnitreeMotorDriver(
            args.sdk_config,
            port0=args.sdk_port0,
            port1=args.sdk_port1,
            baudrate=args.sdk_baudrate,
            timeout=args.sdk_timeout,
        )

    fixed_controller = FixedTeleopController(
        args.command_mode,
        args.command_vx,
        args.command_vy,
        args.command_yaw,
        args.command_estop,
    )
    if args.command_source == "serial":
        if not args.serial_port:
            raise ValueError("--serial_port is required when --command_source serial")
        command_controller = SerialTeleopController(
            args.serial_port,
            baudrate=args.serial_baud,
            protocol=args.serial_protocol,
            default_mode=args.default_mode,
            stale_timeout=args.serial_timeout,
        )
    elif args.command_source == "udp":
        command_controller = UdpTeleopController(
            args.teleop_udp_port,
            args.cmd_vx_max,
            args.cmd_vy_max,
            args.cmd_yaw_max,
        )
    elif args.command_source == "keyboard":
        command_controller = KeyboardTeleopController(args)
    elif args.command_source == "mux":
        command_controller = CommandMux(
            [
                UdpTeleopController(
                    args.teleop_udp_port,
                    args.cmd_vx_max,
                    args.cmd_vy_max,
                    args.cmd_yaw_max,
                ),
                KeyboardTeleopController(args),
                fixed_controller,
            ]
        )
    else:
        command_controller = fixed_controller

    print("[init] Creating MuJoCo environment (this may take a moment)...", flush=True)
    env = MujocoHeightmapSerialEnv(
        xml_path=mujoco_xml,
        cfg=cfg,
        policy=policy,
        device=device,
        control_dt=args.control_dt,
        n_points=n_points,
        hidden_size=hidden_size,
        serial_controller=command_controller,
        height_noise_std=args.height_noise_std,
        motor_backend=args.motor_backend,
        sdk_motor=sdk_motor,
        debug_items=parse_debug_items(args.debug_items),
        debug_every=args.debug_every,
        debug_height_count=args.debug_height_count,
        visualize_heightmap=args.visualize_heightmap,
        heightmap_marker_size=args.heightmap_marker_size,
        visualize_goals=args.visualize_goals or args.policy_type in ("student", "teacher"),
        goal_marker_size=args.goal_marker_size,
        visualize_goal_dirs=args.visualize_goal_dirs or args.visualize_goals or args.policy_type in ("student", "teacher"),
        goal_dir_marker_size=args.goal_dir_marker_size,
        external_goals=None if terrain is None else terrain["goals"],
        input_info_path=input_info_path,
        mode=args.mode,
        policy_type=args.policy_type,
        realtime=args.realtime,
        render=not args.no_render,
    )
    print("[init] Environment ready! Starting simulation...", flush=True)

    viewer_module = maybe_import_viewer(not args.no_render)
    step_count = 0
    last_print_time = time.monotonic()
    try:
        if viewer_module is None:
            print("[sim] Running headless simulation (no rendering)", flush=True)
            while True:
                start = time.monotonic()
                env.step_policy()
                step_count += 1
                
                # Print progress every 1 second
                if time.monotonic() - last_print_time > 1.0:
                    elapsed = time.monotonic() - last_print_time
                    step_rate = step_count / elapsed
                    print(f"[sim] Steps: {step_count}, Speed: {step_rate:.1f} steps/sec", flush=True)
                    step_count = 0
                    last_print_time = time.monotonic()
                
                if args.realtime:
                    time.sleep(max(0.0, args.control_dt - (time.monotonic() - start)))
        else:
            print("[sim] Running with viewer", flush=True)
            with viewer_module.launch_passive(env.model, env.data) as viewer:
                while viewer.is_running():
                    start = time.monotonic()
                    env.step_policy()
                    env.update_heightmap_markers(viewer)
                    viewer.sync()
                    step_count += 1
                    
                    # Print progress every 1 second
                    if time.monotonic() - last_print_time > 1.0:
                        elapsed = time.monotonic() - last_print_time
                        step_rate = step_count / elapsed
                        print(f"[sim] Steps: {step_count}, Speed: {step_rate:.1f} steps/sec", flush=True)
                        step_count = 0
                        last_print_time = time.monotonic()
                    
                    if args.realtime:
                        time.sleep(max(0.0, args.control_dt - (time.monotonic() - start)))
    finally:
        print("[sim] Shutting down...", flush=True)
        # 保存最终的高度图历史
        env.save_heightmap_history()
        print(f"[sim] Total steps recorded: {env.loop_count}, Heightmap history entries: {len(env.heightmap_history)}", flush=True)
        if sdk_motor is not None:
            sdk_motor.set_zero_torque()
            sdk_motor.close()
        command_controller.close()
        print("[sim] Done!", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="traced")
    parser.add_argument("--task_config", type=str, default="mybot_v3", choices=["mybot_v3", "traced"])
    parser.add_argument("--policy_type", type=str, default="student", choices=["student", "teacher"])
    parser.add_argument("--heightmap_model", type=str, default="heightmap_jit.pt")
    parser.add_argument("--teacher_model", type=str, default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--mujoco_xml", type=str, default=MYBOT_V3_XML)
    parser.add_argument("--mujoco_terrain", type=str, default="flat", choices=["flat", "parkour"])
    parser.add_argument("--normalize_mujoco_xml", dest="normalize_mujoco_xml", action="store_true", default=False)
    parser.add_argument("--no_normalize_mujoco_xml", dest="normalize_mujoco_xml", action="store_false")
    parser.add_argument("--terrain_seed", type=int, default=1)
    parser.add_argument("--terrain_difficulty", type=float, default=-1.0, help="Negative samples play.py max_difficulty range [0.7, 1.0].")
    parser.add_argument("--command_source", type=str, default="fixed", choices=["fixed", "keyboard", "udp", "serial", "mux"])
    parser.add_argument("--command_mode", type=int, default=2)
    parser.add_argument("--command_vx", type=float, default=0.2)
    parser.add_argument("--command_vy", type=float, default=0.0)
    parser.add_argument("--command_yaw", type=float, default=0.0)
    parser.add_argument("--command_estop", action="store_true", default=False)
    parser.add_argument("--keyboard_initial_mode", type=int, default=0)
    parser.add_argument("--teleop_udp_port", type=int, default=9870)
    parser.add_argument("--cmd_vx_min", type=float, default=-0.6)
    parser.add_argument("--cmd_vx_max", type=float, default=0.6)
    parser.add_argument("--cmd_vy_min", type=float, default=-0.6)
    parser.add_argument("--cmd_vy_max", type=float, default=0.6)
    parser.add_argument("--cmd_yaw_min", type=float, default=-1.0)
    parser.add_argument("--cmd_yaw_max", type=float, default=1.0)
    parser.add_argument("--cmd_vx_step", type=float, default=0.1)
    parser.add_argument("--cmd_vy_step", type=float, default=0.1)
    parser.add_argument("--cmd_yaw_step", type=float, default=0.2)
    parser.add_argument("--serial_port", type=str, default=None)
    parser.add_argument("--serial_baud", type=int, default=115200)
    parser.add_argument("--serial_protocol", type=str, default="text", choices=["text", "binary"])
    parser.add_argument("--serial_timeout", type=float, default=0.5)
    parser.add_argument("--default_mode", type=int, default=2, help="Mode used when text serial lines only contain vx vy yaw.")
    parser.add_argument("--n_points", type=int, default=None)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--height_noise_std", type=float, default=None)
    parser.add_argument("--motor_backend", type=str, default="mujoco", choices=["mujoco", "sdk", "both"])
    parser.add_argument("--sdk_config", type=str, default=ELMAP_SDK_CONFIG)
    parser.add_argument("--sdk_port0", type=str, default=None)
    parser.add_argument("--sdk_port1", type=str, default=None)
    parser.add_argument("--sdk_baudrate", type=int, default=4000000)
    parser.add_argument("--sdk_timeout", type=float, default=0.02)
    parser.add_argument("--debug_items", type=str, default="", help="Comma list: cmd,base,height,obs,action,target,joint,motor,goal,teacher,all")
    parser.add_argument("--debug_every", type=int, default=0, help="Print debug every N control steps; 0 disables terminal debug.")
    parser.add_argument("--debug_height_count", type=int, default=12)
    parser.add_argument("--visualize_heightmap", action="store_true", default=False)
    parser.add_argument("--heightmap_marker_size", type=float, default=0.025)
    parser.add_argument("--visualize_goals", action="store_true", default=False)
    parser.add_argument("--goal_marker_size", type=float, default=0.08)
    parser.add_argument("--visualize_goal_dirs", action="store_true", default=False)
    parser.add_argument("--goal_dir_marker_size", type=float, default=0.025)
    parser.add_argument("--input_info_path", type=str, default="auto", help="Write first policy input dimensions to this JSON path; empty disables.")
    parser.add_argument("--control_dt", type=float, default=0.02)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--auto_adapt", action="store_true", default=True, help="Auto-wrap loaded model to match expected policy signature")
    parser.add_argument("--mode", type=str, default="parkour", choices=["parkour", "walk"])
    parser.add_argument("--no_render", action="store_true", default=False)
    parser.add_argument("--realtime", dest="realtime", action="store_true", default=True)
    parser.add_argument("--no_realtime", dest="realtime", action="store_false")
    args = parser.parse_args()
    main(args)
