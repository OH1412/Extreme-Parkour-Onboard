"""Bridge deploy-style joint commands to the local MuJoCo lowcmd format.

Input, matching elmap-rl-controller deploy commit
77902c08a3feeef4f64745599a98e8dd1d9c8f29:

  /mujoco/joint_cmd std_msgs/Float32MultiArray
  [target_dof_pos(12), kp_joint(12), kd_joint(12)]

Output, consumed by run_mujoco_serial_ros_sim.py --interface simple:

  /lowcmd std_msgs/Float32MultiArray
  [q_target(12), dq_target(12), kp(12), kd(12), tau_ff(12)]
"""

import argparse
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


DEFAULT_JOINT_ORDER = [
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


def parse_joint_order(value):
    names = [item.strip() for item in value.split(",") if item.strip()]
    if len(names) != 12:
        raise argparse.ArgumentTypeError("joint order must contain exactly 12 comma-separated names")
    if len(set(names)) != 12:
        raise argparse.ArgumentTypeError("joint order contains duplicate names")
    return names


class JointCmdToMujocoLowCmdBridge(Node):
    def __init__(self, args):
        super().__init__("joint_cmd_to_mujoco_lowcmd_bridge")
        self.args = args
        self.source_joint_order = args.source_joint_order
        self.target_joint_order = args.target_joint_order
        self.reorder_indices = self._build_reorder_indices()
        self.last_msg_time = 0.0

        self.lowcmd_pub = self.create_publisher(Float32MultiArray, args.low_cmd_topic, 1)
        self.jointcmd_sub = self.create_subscription(
            Float32MultiArray,
            args.joint_cmd_topic,
            self._joint_cmd_callback,
            1,
        )
        self.timer = self.create_timer(args.timeout_check_dt, self._timeout_check)
        self.get_logger().info(
            "Joint command bridge ready: "
            f"sub {args.joint_cmd_topic} [target,kp,kd] -> "
            f"pub {args.low_cmd_topic} [q,dq,kp,kd,tau], "
            f"source_order={self.source_joint_order}, target_order={self.target_joint_order}"
        )

    def _build_reorder_indices(self):
        missing = [name for name in self.target_joint_order if name not in self.source_joint_order]
        if missing:
            raise ValueError(f"target joint order names missing from source order: {missing}")
        return np.array([self.source_joint_order.index(name) for name in self.target_joint_order], dtype=np.int64)

    def _joint_cmd_callback(self, msg):
        data = np.asarray(msg.data, dtype=np.float32)
        if data.size != 36:
            self.get_logger().warn(
                f"joint_cmd expects 36 floats [target(12),kp(12),kd(12)], got {data.size}",
                throttle_duration_sec=1,
            )
            return
        q_target = data[0:12][self.reorder_indices]
        kp = data[12:24][self.reorder_indices]
        kd = data[24:36][self.reorder_indices]
        self._publish_lowcmd(q_target, kp, kd)
        self.last_msg_time = time.monotonic()

    def _publish_lowcmd(self, q_target, kp, kd):
        zeros = np.zeros(12, dtype=np.float32)
        out = Float32MultiArray()
        out.data = np.concatenate([q_target, zeros, kp, kd, zeros]).astype(np.float32).tolist()
        self.lowcmd_pub.publish(out)

    def _timeout_check(self):
        if self.args.publish_zero_on_timeout <= 0.0 or self.last_msg_time <= 0.0:
            return
        if time.monotonic() - self.last_msg_time <= self.args.publish_zero_on_timeout:
            return
        zeros = np.zeros(12, dtype=np.float32)
        self._publish_lowcmd(zeros, zeros, zeros)
        self.last_msg_time = 0.0
        self.get_logger().warn("joint_cmd timed out; published zero-torque lowcmd")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint_cmd_topic", type=str, default="/mujoco/joint_cmd")
    parser.add_argument("--low_cmd_topic", type=str, default="/lowcmd")
    parser.add_argument("--source_joint_order", type=parse_joint_order, default=DEFAULT_JOINT_ORDER)
    parser.add_argument("--target_joint_order", type=parse_joint_order, default=DEFAULT_JOINT_ORDER)
    parser.add_argument("--publish_zero_on_timeout", type=float, default=0.2)
    parser.add_argument("--timeout_check_dt", type=float, default=0.05)
    args = parser.parse_args()

    rclpy.init()
    node = JointCmdToMujocoLowCmdBridge(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
