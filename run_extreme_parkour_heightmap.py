import rclpy

import os
import os.path as osp
import json
import time
from collections import OrderedDict

import torch

from unitree_ros2_real import UnitreeRos2Real
from sport_api_constants import *


class Go2HeightmapNode(UnitreeRos2Real):
    def __init__(self, *args, n_points=132, update_interval=5, hidden_size=512, **kwargs):
        super().__init__(*args, robot_class_name="Go2", **kwargs)
        self.global_counter = 0
        self.update_interval = update_interval

        self.n_scan = n_points
        self.n_priv_explicit = 3 + 3 + 3
        self.n_priv_latent = 4 + 1 + 12 + 12

        self.hidden_size = hidden_size
        self.policy_hidden = torch.zeros(1, 1, self.hidden_size, device=self.model_device)

        self.use_stand_policy = False
        self.use_parkour_policy = False
        self.use_sport_mode = True

    def reset_obs(self):
        super().reset_obs()
        self.policy_hidden = torch.zeros(1, 1, self.hidden_size, device=self.model_device)

    def register_models(self, turn_obs, policy):
        self.turn_obs = turn_obs
        self.policy = policy

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
        if self.use_sport_mode:
            if self.joy_stick_buffer.keys & self.WirelessButtons.R1:
                self.get_logger().info("In the sport mode, R1 pressed, robot will stand up.")
                self._sport_mode_change(ROBOT_SPORT_API_ID_STANDUP)
            if self.joy_stick_buffer.keys & self.WirelessButtons.R2:
                self.get_logger().info("In the sport mode, R2 pressed, robot will sit down.")
                self._sport_mode_change(ROBOT_SPORT_API_ID_STANDDOWN)
            if self.joy_stick_buffer.keys & self.WirelessButtons.X:
                self.get_logger().info("In the sport mode, X pressed, robot will balance stand.")
                self._sport_mode_change(ROBOT_SPORT_API_ID_BALANCESTAND)
            if self.joy_stick_buffer.keys & self.WirelessButtons.L1:
                self.get_logger().info("Exit sport mode. Switch to stand policy.")
                self.use_sport_mode = False
                self._sport_state_change(0)
                self.use_stand_policy = True
                self.use_parkour_policy = False

        if self.use_stand_policy:
            stand_action = self.get_stand_action()
            self.send_stand_action(stand_action)

        if self.joy_stick_buffer.keys & self.WirelessButtons.Y:
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

        if self.joy_stick_buffer.keys & self.WirelessButtons.L2:
            self.get_logger().info("L2 pressed, stop parkour policy, switch back to sport mode.")
            self.use_stand_policy = False
            self.use_parkour_policy = False
            self.use_sport_mode = True
            self.reset_obs()
            self._sport_state_change(1)
            self._sport_mode_change(ROBOT_SPORT_API_ID_BALANCESTAND)


@torch.inference_mode()
def main(args):
    rclpy.init()

    assert args.logdir is not None, "Please provide a logdir"
    with open(osp.join(args.logdir, "config.json"), "r") as f:
        config_dict = json.load(f, object_pairs_hook=OrderedDict)

    config_dict["control"]["computer_clip_torque"] = True

    device = args.device
    duration = args.control_dt

    env_node = Go2HeightmapNode(
        "go2",
        cfg=config_dict,
        model_device=device,
        dryrun=not args.nodryrun,
        mode=args.mode,
        depth_data_topic=args.heightmap_topic,
        depth_data_shape=[args.n_points],
        n_points=args.n_points,
        update_interval=args.update_interval,
        hidden_size=args.hidden_size,
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
        priv_explicit_zeros = torch.zeros(batch_size, env_node.n_priv_explicit, device=device, dtype=torch.float32)
        priv_latent_zeros = torch.zeros(batch_size, env_node.n_priv_latent, device=device, dtype=torch.float32)
        history_flat = proprio_history.view(batch_size, -1)
        obs = torch.cat(
            [proprio, scan_zeros, priv_explicit_zeros, priv_latent_zeros, history_flat],
            dim=-1,
        )
        return obs

    def actor_model(obs, heightmap_points, hidden):
        return policy(obs, heightmap_points, hidden)

    env_node.register_models(turn_obs=turn_obs, policy=actor_model)

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

    rclpy.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default=None, help="Directory containing config.json and heightmap JIT model")
    parser.add_argument("--heightmap_model", type=str, default="heightmap_jit.pt", help="Heightmap JIT model file name under logdir")
    parser.add_argument("--heightmap_topic", type=str, default="/heightmap_points", help="ROS topic carrying flattened heightmap points")
    parser.add_argument("--n_points", type=int, default=132, help="Number of heightmap points expected from topic")
    parser.add_argument("--hidden_size", type=int, default=512, help="Hidden state size used in traced heightmap encoder")
    parser.add_argument("--update_interval", type=int, default=5, help="Heightmap refresh interval in control iterations")
    parser.add_argument("--control_dt", type=float, default=0.02, help="Control loop period in seconds")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device for inference")
    parser.add_argument("--nodryrun", action="store_true", default=False, help="Disable dryrun mode and send motor commands")
    parser.add_argument("--loop_mode", type=str, default="timer", choices=["while", "timer"], help="Main loop driving mode")
    parser.add_argument("--mode", type=str, default="parkour", choices=["parkour", "walk"])
    args = parser.parse_args()

    main(args)
