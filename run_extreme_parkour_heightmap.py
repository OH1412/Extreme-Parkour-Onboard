import rclpy

import os
import os.path as osp
import json
import select
import sys
import termios
import time
import tty
from collections import OrderedDict

import torch

from std_msgs.msg import Float32MultiArray
from run_mujoco_heightmap_serial import MYBOT_V3_XML, DEPLOY_DOF_NAMES, apply_task_config_overrides, get_xml_hinge_joint_names
from unitree_ros2_real import UnitreeRos2Real
from sport_api_constants import *


class KeyboardController:
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
        self.cmd = {
            "mode": int(args.keyboard_initial_mode),
            "vx": float(args.command_vx),
            "vy": float(args.command_vy),
            "yaw": float(args.command_yaw),
            "e_stop": False,
        }
        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd) if sys.stdin.isatty() else None
        if self._old_settings is not None:
            tty.setcbreak(self._fd)

    def close(self):
        if self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
            self._old_settings = None

    def get_latest(self):
        self._poll()
        return dict(self.cmd)

    def _poll(self):
        while True:
            readable, _, _ = select.select([sys.stdin], [], [], 0.0)
            if not readable:
                return
            key = sys.stdin.read(1)
            if key:
                self._process_key(key)

    def _process_key(self, key):
        if key in ("w", "W"):
            self.cmd["vx"] = min(self.cmd["vx"] + self.cmd_vx_step, self.cmd_vx_max)
        elif key in ("s", "S"):
            self.cmd["vx"] = max(self.cmd["vx"] - self.cmd_vx_step, self.cmd_vx_min)
        elif key in ("q", "Q"):
            self.cmd["vy"] = min(self.cmd["vy"] + self.cmd_vy_step, self.cmd_vy_max)
        elif key in ("e", "E"):
            self.cmd["vy"] = max(self.cmd["vy"] - self.cmd_vy_step, self.cmd_vy_min)
        elif key in ("a", "A"):
            self.cmd["yaw"] = min(self.cmd["yaw"] + self.cmd_yaw_step, self.cmd_yaw_max)
        elif key in ("d", "D"):
            self.cmd["yaw"] = max(self.cmd["yaw"] - self.cmd_yaw_step, self.cmd_yaw_min)
        elif key == "0":
            self.cmd["mode"] = 0
            self._zero_commands()
        elif key == "1":
            self.cmd["mode"] = 1
            self._zero_commands()
        elif key == "2":
            self.cmd["mode"] = 2
            self.cmd["e_stop"] = False
        elif key == "3":
            self.cmd["mode"] = 3
            self._zero_commands()
        elif key in ("r", "R"):
            self._zero_commands()
        elif key == " ":
            self.cmd["mode"] = 0
            self.cmd["e_stop"] = True
            self._zero_commands()
        print(
            "keyboard cmd "
            f"mode:{self.cmd['mode']} vx:{self.cmd['vx']:.2f} "
            f"vy:{self.cmd['vy']:.2f} yaw:{self.cmd['yaw']:.2f} "
            f"estop:{int(self.cmd['e_stop'])}",
            flush=True,
        )

    def _zero_commands(self):
        self.cmd["vx"] = 0.0
        self.cmd["vy"] = 0.0
        self.cmd["yaw"] = 0.0
        self.cmd["e_stop"] = False


class Go2HeightmapNode(UnitreeRos2Real):
    def __init__(
        self,
        *args,
        n_points=132,
        update_interval=5,
        hidden_size=512,
        command_topic="/parkour/command",
        goal_yaw_topic="/parkour/goal_yaw",
        mode_flags_topic="/parkour/mode_flags",
        foot_contacts_topic="/parkour/foot_contacts",
        priv_explicit_topic="/parkour/priv_explicit",
        priv_latent_topic="/parkour/priv_latent",
        use_external_command=False,
        use_external_goal_yaw=True,
        use_external_mode_flags=False,
        use_external_foot_contacts=False,
        use_external_priv_explicit=False,
        use_external_priv_latent=False,
        keyboard_controller=None,
        **kwargs,
    ):
        super().__init__(*args, robot_class_name="Go2", **kwargs)
        self.global_counter = 0
        self.update_interval = update_interval

        self.n_scan = n_points
        self.n_priv_explicit = 3 + 3 + 3
        self.n_priv_latent = 4 + 1 + 12 + 12

        self.hidden_size = hidden_size
        self.policy_hidden = torch.zeros(1, 1, self.hidden_size, device=self.model_device)

        self.command_topic = command_topic
        self.goal_yaw_topic = goal_yaw_topic
        self.mode_flags_topic = mode_flags_topic
        self.foot_contacts_topic = foot_contacts_topic
        self.priv_explicit_topic = priv_explicit_topic
        self.priv_latent_topic = priv_latent_topic

        self.use_external_command = use_external_command
        self.use_external_goal_yaw = use_external_goal_yaw
        self.use_external_mode_flags = use_external_mode_flags
        self.use_external_foot_contacts = use_external_foot_contacts
        self.use_external_priv_explicit = use_external_priv_explicit
        self.use_external_priv_latent = use_external_priv_latent
        self.keyboard_controller = keyboard_controller
        self.last_keyboard_mode = None

        self.external_goal_yaw = torch.zeros(1, 3, device=self.model_device, dtype=torch.float32)
        self.external_mode_flags = torch.tensor([[1.0, 0.0]], device=self.model_device, dtype=torch.float32)
        self.external_foot_contacts = torch.zeros(1, 4, device=self.model_device, dtype=torch.float32)
        self.external_priv_explicit = torch.zeros(1, self.n_priv_explicit, device=self.model_device, dtype=torch.float32)
        self.external_priv_latent = torch.zeros(1, self.n_priv_latent, device=self.model_device, dtype=torch.float32)

        self.use_stand_policy = False
        self.use_parkour_policy = False
        self.use_sport_mode = True

    def reset_obs(self):
        super().reset_obs()
        self.policy_hidden = torch.zeros(1, 1, self.hidden_size, device=self.model_device)
        self.external_goal_yaw = torch.zeros(1, 3, device=self.model_device, dtype=torch.float32)

    def start_ros_handlers(self):
        super().start_ros_handlers()
        self._start_external_obs_subscribers()

    def _start_external_obs_subscribers(self):
        if self.use_external_command:
            self.external_command_sub = self.create_subscription(
                Float32MultiArray,
                self.command_topic,
                self._external_command_callback,
                1,
            )
            self.get_logger().info(f"External command subscriber: {self.command_topic}")
        if self.use_external_goal_yaw:
            self.external_goal_yaw_sub = self.create_subscription(
                Float32MultiArray,
                self.goal_yaw_topic,
                self._external_goal_yaw_callback,
                1,
            )
            self.get_logger().info(f"External goal yaw subscriber: {self.goal_yaw_topic}")
        if self.use_external_mode_flags:
            self.external_mode_flags_sub = self.create_subscription(
                Float32MultiArray,
                self.mode_flags_topic,
                self._external_mode_flags_callback,
                1,
            )
            self.get_logger().info(f"External mode flags subscriber: {self.mode_flags_topic}")
        if self.use_external_foot_contacts:
            self.external_foot_contacts_sub = self.create_subscription(
                Float32MultiArray,
                self.foot_contacts_topic,
                self._external_foot_contacts_callback,
                1,
            )
            self.get_logger().info(f"External foot contacts subscriber: {self.foot_contacts_topic}")
        if self.use_external_priv_explicit:
            self.external_priv_explicit_sub = self.create_subscription(
                Float32MultiArray,
                self.priv_explicit_topic,
                self._external_priv_explicit_callback,
                1,
            )
            self.get_logger().info(f"External priv explicit subscriber: {self.priv_explicit_topic}")
        if self.use_external_priv_latent:
            self.external_priv_latent_sub = self.create_subscription(
                Float32MultiArray,
                self.priv_latent_topic,
                self._external_priv_latent_callback,
                1,
            )
            self.get_logger().info(f"External priv latent subscriber: {self.priv_latent_topic}")

    def _tensor_from_msg(self, msg, expected_len, name):
        data = torch.tensor(msg.data, dtype=torch.float32, device=self.model_device)
        if data.numel() != expected_len:
            self.get_logger().warn(
                f"{name} topic shape mismatch: got {data.numel()} values, expected {expected_len}.",
                throttle_duration_sec=1,
            )
            return None
        return data.view(1, expected_len)

    def _external_command_callback(self, msg):
        data = self._tensor_from_msg(msg, 3, "External command")
        if data is not None:
            self.xyyaw_command = data

    def _external_goal_yaw_callback(self, msg):
        data = torch.tensor(msg.data, dtype=torch.float32, device=self.model_device)
        if data.numel() == 2:
            self.external_goal_yaw = torch.tensor(
                [[0.0, data[0].item(), data[1].item()]],
                dtype=torch.float32,
                device=self.model_device,
            )
        elif data.numel() == 3:
            self.external_goal_yaw = data.view(1, 3)
        else:
            self.get_logger().warn(
                f"External goal yaw topic shape mismatch: got {data.numel()} values, expected 2 or 3.",
                throttle_duration_sec=1,
            )

    def _external_mode_flags_callback(self, msg):
        data = self._tensor_from_msg(msg, 2, "External mode flags")
        if data is not None:
            self.external_mode_flags = data

    def _external_foot_contacts_callback(self, msg):
        data = self._tensor_from_msg(msg, 4, "External foot contacts")
        if data is not None:
            self.external_foot_contacts = torch.where(
                data >= 0.5,
                torch.full_like(data, 0.5),
                torch.full_like(data, -0.5),
            )

    def _external_priv_explicit_callback(self, msg):
        data = self._tensor_from_msg(msg, self.n_priv_explicit, "External priv explicit")
        if data is not None:
            self.external_priv_explicit = data

    def _external_priv_latent_callback(self, msg):
        data = self._tensor_from_msg(msg, self.n_priv_latent, "External priv latent")
        if data is not None:
            self.external_priv_latent = data

    def _get_delta_yaw_obs(self):
        if self.use_external_goal_yaw:
            return self.external_goal_yaw
        return super()._get_delta_yaw_obs()

    def _get_commands_obs(self):
        if self.use_external_command:
            vx, _, _ = self.xyyaw_command[0, :]
            return torch.tensor([[0.0, 0.0, vx.item()]], device=self.model_device, dtype=torch.float32)
        return super()._get_commands_obs()

    def _get_contact_filt_obs(self):
        if self.use_external_foot_contacts:
            return self.external_foot_contacts
        return super()._get_contact_filt_obs()

    def get_proprio(self):
        proprio = super().get_proprio()
        if self.use_external_mode_flags:
            proprio[:, 11:13] = self.external_mode_flags
        return proprio

    def register_models(self, turn_obs, policy):
        self.turn_obs = turn_obs
        self.policy = policy

    def _joy_keys(self):
        if getattr(self, "ros_interface", "unitree") == "simple":
            return getattr(self, "simple_joy_keys", 0)
        return self.joy_stick_buffer.keys

    def start_main_loop_timer(self, duration):
        self.main_loop_timer = self.create_timer(duration, self.main_loop)

    def warm_up(self):
        if not hasattr(self, "depth_data"):
            self.get_logger().warn("Heightmap points not received yet; skip warm up.", once=True)
            return

        for _ in range(2):
            proprio = self.get_proprio()
            proprio_history = self._get_history_proprio()
            heightmap_points = self._get_depth_image()
            obs = self.turn_obs(proprio, proprio_history)
            _, self.policy_hidden = self.policy(obs, heightmap_points, self.policy_hidden)

    def main_loop(self):
        keyboard_active = self.keyboard_controller is not None
        if keyboard_active:
            self._apply_keyboard_control()
            joy_keys = 0
        else:
            joy_keys = self._joy_keys()

        if not keyboard_active and self.use_sport_mode:
            if joy_keys & self.WirelessButtons.R1:
                self.get_logger().info("In the sport mode, R1 pressed, robot will stand up.")
                self._sport_mode_change(ROBOT_SPORT_API_ID_STANDUP)
            if joy_keys & self.WirelessButtons.R2:
                self.get_logger().info("In the sport mode, R2 pressed, robot will sit down.")
                self._sport_mode_change(ROBOT_SPORT_API_ID_STANDDOWN)
            if joy_keys & self.WirelessButtons.X:
                self.get_logger().info("In the sport mode, X pressed, robot will balance stand.")
                self._sport_mode_change(ROBOT_SPORT_API_ID_BALANCESTAND)
            if joy_keys & self.WirelessButtons.L1:
                self.get_logger().info("Exit sport mode. Switch to stand policy.")
                self.use_sport_mode = False
                self._sport_state_change(0)
                self.use_stand_policy = True
                self.use_parkour_policy = False

        if self.use_stand_policy:
            stand_action = self.get_stand_action()
            self.send_stand_action(stand_action)

        if not keyboard_active and joy_keys & self.WirelessButtons.Y:
            self.get_logger().info("Y pressed, use the parkour policy")
            self.use_stand_policy = False
            self.use_parkour_policy = True
            self.use_sport_mode = False
            self.global_counter = 0
            self.policy_hidden = torch.zeros(1, 1, self.hidden_size, device=self.model_device)

        if self.use_parkour_policy:
            self.use_stand_policy = False
            self.use_sport_mode = False

            proprio = self.get_proprio()
            proprio_history = self._get_history_proprio()

            if self.global_counter % self.update_interval == 0:
                if hasattr(self, "depth_data"):
                    self.last_heightmap = self._get_depth_image()
                elif not hasattr(self, "last_heightmap"):
                    self.get_logger().warn(
                        "Waiting for heightmap points on subscribed topic.",
                        throttle_duration_sec=1,
                    )
                    self.global_counter += 1
                    return

            obs = self.turn_obs(proprio, proprio_history)
            action, self.policy_hidden = self.policy(obs, self.last_heightmap, self.policy_hidden)
            self.send_action(action)
            self.global_counter += 1

        if not keyboard_active and joy_keys & self.WirelessButtons.L2:
            self.get_logger().info("L2 pressed, stop parkour policy, switch back to sport mode.")
            self.use_stand_policy = False
            self.use_parkour_policy = False
            self.use_sport_mode = True
            self.reset_obs()
            self._sport_state_change(1)
            self._sport_mode_change(ROBOT_SPORT_API_ID_BALANCESTAND)

    def _apply_keyboard_control(self):
        cmd = self.keyboard_controller.get_latest()
        self.xyyaw_command = torch.tensor(
            [[cmd["vx"], cmd["vy"], cmd["yaw"]]],
            device=self.model_device,
            dtype=torch.float32,
        )
        mode = int(cmd["mode"])
        if cmd["e_stop"] or mode == 0:
            self.use_stand_policy = False
            self.use_parkour_policy = False
            self.use_sport_mode = True
            self._turn_off_motors()
        elif mode == 1 or mode == 4:
            self.use_stand_policy = True
            self.use_parkour_policy = False
            self.use_sport_mode = False
        elif mode == 2:
            if self.last_keyboard_mode != 2:
                self.get_logger().info("Keyboard mode 2, use the parkour policy")
                self.global_counter = 0
                self.policy_hidden = torch.zeros(1, 1, self.hidden_size, device=self.model_device)
            self.use_stand_policy = False
            self.use_parkour_policy = True
            self.use_sport_mode = False
        elif mode == 3:
            self.use_stand_policy = False
            self.use_parkour_policy = False
            self.use_sport_mode = True
            self._turn_off_motors()
        self.last_keyboard_mode = mode


@torch.inference_mode()
def main(args):
    rclpy.init()

    assert args.logdir is not None, "Please provide a logdir"
    with open(osp.join(args.logdir, "config.json"), "r") as f:
        config_dict = json.load(f, object_pairs_hook=OrderedDict)
    config_dict = apply_task_config_overrides(config_dict, args.task_config)

    config_dict["control"]["computer_clip_torque"] = True
    simple_dof_names = None
    if args.interface in ("auto", "simple"):
        if bool(config_dict.get("env", {}).get("reorder_dofs", True)):
            simple_dof_names = DEPLOY_DOF_NAMES
        else:
            import mujoco

            simple_dof_names = get_xml_hinge_joint_names(mujoco.MjModel.from_xml_path(args.mujoco_xml))

    device = args.device
    duration = args.control_dt
    keyboard_controller = KeyboardController(args) if args.keyboard_control else None

    env_node = Go2HeightmapNode(
        "go2",
        low_state_topic=args.low_state_topic,
        low_cmd_topic=args.low_cmd_topic,
        joy_stick_topic=args.joy_stick_topic,
        cfg=config_dict,
        model_device=device,
        dryrun=not args.nodryrun,
        dryrun_lowcmd_suffix=not args.no_lowcmd_dryrun_suffix,
        ros_interface=args.interface,
        require_joy_stick=not args.keyboard_control,
        simple_dof_names=simple_dof_names,
        mode=args.mode,
        depth_data_topic=args.heightmap_topic,
        depth_data_shape=[args.n_points],
        n_points=args.n_points,
        update_interval=args.update_interval,
        hidden_size=args.hidden_size,
        command_topic=args.command_topic,
        goal_yaw_topic=args.goal_yaw_topic,
        mode_flags_topic=args.mode_flags_topic,
        foot_contacts_topic=args.foot_contacts_topic,
        priv_explicit_topic=args.priv_explicit_topic,
        priv_latent_topic=args.priv_latent_topic,
        use_external_command=args.use_external_command,
        use_external_goal_yaw=not args.disable_external_goal_yaw,
        use_external_mode_flags=args.use_external_mode_flags,
        use_external_foot_contacts=args.use_external_foot_contacts,
        use_external_priv_explicit=args.use_external_priv_explicit,
        use_external_priv_latent=args.use_external_priv_latent,
        keyboard_controller=keyboard_controller,
    )

    env_node.get_logger().info("Control Duration: {} sec".format(duration))
    env_node.get_logger().info("Motor Stiffness (kp): {}".format(env_node.p_gains))
    env_node.get_logger().info("Motor Damping (kd): {}".format(env_node.d_gains))

    model_path = os.path.join(args.logdir, args.heightmap_model)
    policy = torch.jit.load(model_path, map_location=device)
    policy.eval()

    env_node.get_logger().info("Heightmap JIT model loaded from: {}".format(model_path))

    def turn_obs(proprio, proprio_history):
        batch_size = proprio.shape[0]
        scan_zeros = torch.zeros(batch_size, env_node.n_scan, device=device, dtype=torch.float32)
        priv_explicit = env_node.external_priv_explicit if env_node.use_external_priv_explicit else torch.zeros(
            batch_size,
            env_node.n_priv_explicit,
            device=device,
            dtype=torch.float32,
        )
        priv_latent = env_node.external_priv_latent if env_node.use_external_priv_latent else torch.zeros(
            batch_size,
            env_node.n_priv_latent,
            device=device,
            dtype=torch.float32,
        )
        history_flat = proprio_history.view(batch_size, -1)
        obs = torch.cat(
            [proprio, scan_zeros, priv_explicit, priv_latent, history_flat],
            dim=-1,
        )
        return obs

    def actor_model(obs, heightmap_points, hidden):
        return policy(obs, heightmap_points, hidden)

    env_node.register_models(turn_obs=turn_obs, policy=actor_model)

    try:
        env_node.start_ros_handlers()
        env_node.warm_up()

        if args.loop_mode == "while":
            rclpy.spin_once(env_node, timeout_sec=0.0)
            env_node.get_logger().info("Model and Policy are ready")
            while rclpy.ok():
                main_loop_time = time.monotonic()
                env_node.main_loop()
                rclpy.spin_once(env_node, timeout_sec=0.0)
                time.sleep(max(0, duration - (time.monotonic() - main_loop_time)))
        elif args.loop_mode == "timer":
            env_node.get_logger().info("Model and Policy are ready")
            env_node.start_main_loop_timer(duration)
            rclpy.spin(env_node)
    finally:
        if keyboard_controller is not None:
            keyboard_controller.close()
        rclpy.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default=None, help="Directory containing config.json and heightmap JIT model")
    parser.add_argument("--task_config", type=str, default="mybot_v3", choices=["mybot_v3", "traced"])
    parser.add_argument("--mujoco_xml", type=str, default=MYBOT_V3_XML)
    parser.add_argument("--heightmap_model", type=str, default="heightmap_jit.pt", help="Heightmap JIT model file name under logdir")
    parser.add_argument("--low_state_topic", type=str, default="/lowstate", help="LowState input topic")
    parser.add_argument("--low_cmd_topic", type=str, default="/lowcmd", help="LowCmd output topic")
    parser.add_argument("--joy_stick_topic", type=str, default="/wirelesscontroller", help="WirelessController input topic")
    parser.add_argument("--interface", type=str, default="auto", choices=["auto", "simple", "unitree"], help="ROS message interface for lowstate/lowcmd/wireless")
    parser.add_argument("--heightmap_topic", type=str, default="/parkour/heightmap_points", help="Float32MultiArray, length n_points")
    parser.add_argument("--command_topic", type=str, default="/parkour/command", help="Float32MultiArray [vx, vy, yaw]")
    parser.add_argument("--goal_yaw_topic", type=str, default="/parkour/goal_yaw", help="Float32MultiArray [delta_yaw, delta_next_yaw] or [0, delta_yaw, delta_next_yaw]")
    parser.add_argument("--mode_flags_topic", type=str, default="/parkour/mode_flags", help="Float32MultiArray [parkour_flag, walk_flag]")
    parser.add_argument("--foot_contacts_topic", type=str, default="/parkour/foot_contacts", help="Float32MultiArray length 4, raw 0/1 contacts")
    parser.add_argument("--priv_explicit_topic", type=str, default="/parkour/priv_explicit", help="Float32MultiArray length 9")
    parser.add_argument("--priv_latent_topic", type=str, default="/parkour/priv_latent", help="Float32MultiArray length 29")
    parser.add_argument("--use_external_command", action="store_true", default=False, help="Use command_topic instead of wireless controller command")
    parser.add_argument("--disable_external_goal_yaw", action="store_true", default=False, help="Keep goal yaw observation at zero instead of subscribing goal_yaw_topic")
    parser.add_argument("--use_external_mode_flags", action="store_true", default=False, help="Use mode_flags_topic instead of --mode")
    parser.add_argument("--use_external_foot_contacts", action="store_true", default=False, help="Use foot_contacts_topic instead of lowstate foot force contacts")
    parser.add_argument("--use_external_priv_explicit", action="store_true", default=False, help="Use priv_explicit_topic in obs instead of zeros")
    parser.add_argument("--use_external_priv_latent", action="store_true", default=False, help="Use priv_latent_topic in obs instead of zeros")
    parser.add_argument("--n_points", type=int, default=132, help="Number of heightmap points expected from topic")
    parser.add_argument("--hidden_size", type=int, default=512, help="Hidden state size used in traced heightmap encoder")
    parser.add_argument("--update_interval", type=int, default=5, help="Heightmap refresh interval in control iterations")
    parser.add_argument("--control_dt", type=float, default=0.02, help="Control loop period in seconds")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device for inference")
    parser.add_argument("--nodryrun", action="store_true", default=False, help="Disable dryrun mode and send motor commands")
    parser.add_argument("--no_lowcmd_dryrun_suffix", action="store_true", default=False, help="Keep low_cmd_topic unchanged while dryrun is enabled")
    parser.add_argument("--keyboard_control", action="store_true", default=False, help="Read keyboard commands directly in this controller process")
    parser.add_argument("--keyboard_initial_mode", type=int, default=2)
    parser.add_argument("--command_vx", type=float, default=0.5)
    parser.add_argument("--command_vy", type=float, default=0.0)
    parser.add_argument("--command_yaw", type=float, default=0.0)
    parser.add_argument("--cmd_vx_min", type=float, default=-0.6)
    parser.add_argument("--cmd_vx_max", type=float, default=0.6)
    parser.add_argument("--cmd_vy_min", type=float, default=-0.6)
    parser.add_argument("--cmd_vy_max", type=float, default=0.6)
    parser.add_argument("--cmd_yaw_min", type=float, default=-1.0)
    parser.add_argument("--cmd_yaw_max", type=float, default=1.0)
    parser.add_argument("--cmd_vx_step", type=float, default=0.1)
    parser.add_argument("--cmd_vy_step", type=float, default=0.1)
    parser.add_argument("--cmd_yaw_step", type=float, default=0.2)
    parser.add_argument("--loop_mode", type=str, default="timer", choices=["while", "timer"], help="Main loop driving mode")
    parser.add_argument("--mode", type=str, default="parkour", choices=["parkour", "walk"])
    args = parser.parse_args()

    main(args)
