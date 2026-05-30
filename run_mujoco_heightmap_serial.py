import argparse
import json
import os
import os.path as osp
import select
import socket
import struct
import sys
import termios
import threading
import time
import tty
from collections import OrderedDict

import mujoco
import numpy as np
import torch

from serial_teleop import SerialTeleopController
from unitree_motor_sdk_python import PythonUnitreeMotorDriver


MYBOT_V3_XML = "/home/rc_kfs/extreme-parkour/legged_gym/resources/robots/mybot_v3/xml/mybot_v3.xml"
ELMAP_SDK_CONFIG = "/home/rc_kfs/el_ws/elmap-rl-controller/deploy_cpp/config/robots/mybot_v2_1_cse.yaml"


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


def parse_debug_items(items):
    if not items:
        return set()
    out = set()
    for item in items.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item == "all":
            return {"cmd", "base", "height", "obs", "action", "target", "joint", "motor"}
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
        mode="parkour",
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
        self.mode = mode
        self.realtime = realtime
        self.render = render
        self.rng = np.random.default_rng()

        self.reorder_dofs = bool(cfg.get("env", {}).get("reorder_dofs", True))
        self.include_foot_contacts = bool(cfg.get("env", {}).get("include_foot_contacts", True))
        self.dof_names = DEPLOY_DOF_NAMES if self.reorder_dofs else get_xml_hinge_joint_names(self.model)
        self.num_dof = len(self.dof_names)
        if self.num_dof != int(cfg.get("env", {}).get("num_actions", 12)):
            raise ValueError(
                f"MuJoCo model has {self.num_dof} hinge joints in policy order, "
                f"but config expects {cfg.get('env', {}).get('num_actions', 12)} actions."
            )
        self.n_proprio = int(cfg.get("env", {}).get("n_proprio", 53 if self.include_foot_contacts else 49))
        self.n_hist_len = int(cfg.get("env", {}).get("history_len", 10))
        self.n_priv_explicit = 3 + 3 + 3
        self.n_priv_latent = 4 + 1 + 12 + 12
        self.hidden = torch.zeros(1, 1, hidden_size, device=device)
        self.proprio_history_buf = torch.zeros(1, self.n_hist_len, self.n_proprio, device=device)
        self.episode_length_buf = torch.zeros(1, device=device)
        self.actions = torch.zeros(self.num_dof, device=device)
        self.contact_filt = torch.ones((1, 4), device=device)
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
        self.last_target = None

        self.default_dof_pos = torch.tensor(
            [cfg["init_state"]["default_joint_angles"][name] for name in self.dof_names],
            device=device,
            dtype=torch.float32,
        )
        self.p_gains = self._gains_from_config("stiffness")
        self.d_gains = self._gains_from_config("damping")
        self.action_scale = float(cfg["control"]["action_scale"])
        self.clip_actions = float(cfg["normalization"]["clip_actions"])
        self.obs_scales = cfg["normalization"]["obs_scales"]

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
        mujoco.mj_forward(self.model, self.data)

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
            proprio_history = self.proprio_history_buf
            base_pos, _, yaw = self.get_base_pose()
            heightmap_points = self.sample_heightmap(base_pos, yaw)
            obs = self.turn_obs(proprio, proprio_history)
            self.last_proprio = proprio.detach().cpu()
            self.last_obs = obs.detach().cpu()
            self.last_heightmap = heightmap_points.detach().cpu()
            actions, self.hidden = self.policy(obs, heightmap_points, self.hidden)
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
        yaw_info = torch.zeros(1, 3, device=self.device)
        commands = torch.tensor([[0.0, 0.0, float(cmd["vx"])]], device=self.device)
        parkour_walk = torch.tensor(
            [[1.0, 0.0] if self.mode == "parkour" else [0.0, 1.0]],
            device=self.device,
            dtype=torch.float32,
        )
        dof_pos = (self.current_dof_pos().view(1, -1) - self.default_dof_pos.view(1, -1))
        dof_pos = dof_pos * float(self.obs_scales["dof_pos"])
        dof_vel = self.current_dof_vel().view(1, -1) * float(self.obs_scales["dof_vel"])
        last_actions = self.actions.view(1, -1)
        proprio_parts = [ang_vel, imu, yaw_info, commands, parkour_walk, dof_pos, dof_vel, last_actions]
        if self.include_foot_contacts:
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
        actions = actions.view(-1).to(self.device)
        self.actions = actions
        self.last_action = actions.detach().cpu()
        hard_clip = self.clip_actions / self.action_scale
        target = torch.clip(actions, -hard_clip, hard_clip) * self.action_scale + self.default_dof_pos
        self._apply_pd(target, self.p_gains, self.d_gains)

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
        if "action" in self.debug_items and self.last_action is not None:
            a = self.last_action.view(-1)
            parts.append(f"action=maxabs:{a.abs().max():.3f} first3:{a[:3].numpy()}")
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
        if not self.visualize_heightmap or self.last_heightmap_world_points is None:
            return
        scene = viewer.user_scn
        scene.ngeom = 0
        mat = np.eye(3, dtype=np.float64).reshape(-1)
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

    def _contact_obs(self):
        feet = ["FR", "FL", "RR", "RL"]
        contact = np.full(4, -0.5, dtype=np.float32)
        for i in range(self.data.ncon):
            mj_contact = self.data.contact[i]
            geom1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, mj_contact.geom1) or ""
            geom2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, mj_contact.geom2) or ""
            pair_name = f"{geom1} {geom2}"
            for foot_idx, prefix in enumerate(feet):
                if prefix in pair_name:
                    contact[foot_idx] = 0.5
        return torch.tensor(contact, device=self.device, dtype=torch.float32).view(1, 4)


def maybe_import_viewer(render):
    if not render:
        return None
    import mujoco.viewer

    return mujoco.viewer


@torch.inference_mode()
def main(args):
    with open(osp.join(args.logdir, "config.json"), "r") as f:
        cfg = json.load(f, object_pairs_hook=OrderedDict)
    cfg["control"]["computer_clip_torque"] = True
    n_points = args.n_points
    if n_points is None:
        n_points = int(cfg.get("heightmap", {}).get("n_points", cfg.get("env", {}).get("n_scan", 132)))
    hidden_size = args.hidden_size
    if hidden_size is None:
        hidden_size = int(cfg.get("heightmap_encoder", {}).get("hidden_size", 512))

    device = args.device
    model_path = osp.join(args.logdir, args.heightmap_model)
    policy = torch.jit.load(model_path, map_location=device)
    policy.eval()

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

    env = MujocoHeightmapSerialEnv(
        xml_path=args.mujoco_xml,
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
        mode=args.mode,
        realtime=args.realtime,
        render=not args.no_render,
    )

    viewer_module = maybe_import_viewer(not args.no_render)
    try:
        if viewer_module is None:
            while True:
                start = time.monotonic()
                env.step_policy()
                if args.realtime:
                    time.sleep(max(0.0, args.control_dt - (time.monotonic() - start)))
        else:
            with viewer_module.launch_passive(env.model, env.data) as viewer:
                while viewer.is_running():
                    start = time.monotonic()
                    env.step_policy()
                    env.update_heightmap_markers(viewer)
                    viewer.sync()
                    if args.realtime:
                        time.sleep(max(0.0, args.control_dt - (time.monotonic() - start)))
    finally:
        if sdk_motor is not None:
            sdk_motor.set_zero_torque()
            sdk_motor.close()
        command_controller.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="traced")
    parser.add_argument("--heightmap_model", type=str, default="heightmap_jit.pt")
    parser.add_argument("--mujoco_xml", type=str, default=MYBOT_V3_XML)
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
    parser.add_argument("--debug_items", type=str, default="", help="Comma list: cmd,base,height,obs,action,target,joint,motor,all")
    parser.add_argument("--debug_every", type=int, default=0, help="Print debug every N control steps; 0 disables terminal debug.")
    parser.add_argument("--debug_height_count", type=int, default=12)
    parser.add_argument("--visualize_heightmap", action="store_true", default=False)
    parser.add_argument("--heightmap_marker_size", type=float, default=0.025)
    parser.add_argument("--control_dt", type=float, default=0.02)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--mode", type=str, default="parkour", choices=["parkour", "walk"])
    parser.add_argument("--no_render", action="store_true", default=False)
    parser.add_argument("--realtime", dest="realtime", action="store_true", default=True)
    parser.add_argument("--no_realtime", dest="realtime", action="store_false")
    args = parser.parse_args()
    main(args)
