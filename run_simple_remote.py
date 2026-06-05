"""Publish simple ROS teleop topics for the MuJoCo bridge/controller setup."""

import argparse
import select
import sys
import termios
import time
import tty

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class WirelessButtons:
    R1 = 0b00000001
    L1 = 0b00000010
    R2 = 0b00010000
    L2 = 0b00100000
    X = 0b10000000000
    Y = 0b100000000000


class SimpleRemote(Node):
    def __init__(self, args):
        super().__init__("simple_remote")
        self.args = args
        self.keys = 0
        self.key_until = {}
        self.vx = float(args.vx)
        self.vy = float(args.vy)
        self.yaw = float(args.yaw)
        if args.auto_parkour:
            self._hold_key(WirelessButtons.Y)
        self.wireless_pub = self.create_publisher(Float32MultiArray, args.joy_stick_topic, 1)
        self.command_pub = self.create_publisher(Float32MultiArray, args.command_topic, 1)
        self.timer = self.create_timer(1.0 / args.rate, self.publish)
        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd) if sys.stdin.isatty() else None
        if self._old_settings is not None:
            tty.setcbreak(self._fd)
        self.get_logger().info(
            "Keys: l=L1 stand, y=parkour, x=balance, r=R1 standup, 2=R2 sit, "
            "p=L2 stop, w/s vx, q/e vy, a/d yaw, space zero, Ctrl-C quit"
        )

    def close(self):
        if self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
            self._old_settings = None

    def publish(self):
        self._read_key()
        now = time.monotonic()
        active_keys = 0
        expired = []
        for key_bit, until in self.key_until.items():
            if now <= until:
                active_keys |= key_bit
            else:
                expired.append(key_bit)
        for key_bit in expired:
            self.key_until.pop(key_bit, None)
        joy_msg = Float32MultiArray()
        joy_msg.data = [float(active_keys)]
        self.wireless_pub.publish(joy_msg)

        cmd_msg = Float32MultiArray()
        cmd_msg.data = [float(self.vx), float(self.vy), float(self.yaw)]
        self.command_pub.publish(cmd_msg)

    def _read_key(self):
        while True:
            readable, _, _ = select.select([sys.stdin], [], [], 0.0)
            if not readable:
                return
            key = sys.stdin.read(1)
            if key:
                self._process_key(key)

    def _process_key(self, key):
        if key in ("l", "L"):
            self._hold_key(WirelessButtons.L1)
        elif key in ("y", "Y"):
            self._hold_key(WirelessButtons.Y)
        elif key in ("x", "X"):
            self._hold_key(WirelessButtons.X)
        elif key in ("r", "R"):
            self._hold_key(WirelessButtons.R1)
        elif key == "2":
            self._hold_key(WirelessButtons.R2)
        elif key in ("p", "P"):
            self._hold_key(WirelessButtons.L2)
        elif key in ("w", "W"):
            self.vx = min(self.args.vx_max, self.vx + self.args.vx_step)
        elif key in ("s", "S"):
            self.vx = max(self.args.vx_min, self.vx - self.args.vx_step)
        elif key in ("q", "Q"):
            self.vy = min(self.args.vy_max, self.vy + self.args.vy_step)
        elif key in ("e", "E"):
            self.vy = max(self.args.vy_min, self.vy - self.args.vy_step)
        elif key in ("a", "A"):
            self.yaw = min(self.args.yaw_max, self.yaw + self.args.yaw_step)
        elif key in ("d", "D"):
            self.yaw = max(self.args.yaw_min, self.yaw - self.args.yaw_step)
        elif key == " ":
            self.vx = 0.0
            self.vy = 0.0
            self.yaw = 0.0
        self.get_logger().info(
            f"keys={sum(self.key_until.keys())} cmd=[{self.vx:.2f}, {self.vy:.2f}, {self.yaw:.2f}]",
            throttle_duration_sec=0.2,
        )

    def _hold_key(self, key_bit):
        self.key_until[key_bit] = time.monotonic() + self.args.key_hold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--joy_stick_topic", type=str, default="/wirelesscontroller")
    parser.add_argument("--command_topic", type=str, default="/parkour/command")
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--key_hold", type=float, default=0.25)
    parser.add_argument("--auto_parkour", action="store_true", default=False)
    parser.add_argument("--vx", type=float, default=0.5)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--vx_min", type=float, default=-0.6)
    parser.add_argument("--vx_max", type=float, default=0.6)
    parser.add_argument("--vy_min", type=float, default=-0.6)
    parser.add_argument("--vy_max", type=float, default=0.6)
    parser.add_argument("--yaw_min", type=float, default=-1.0)
    parser.add_argument("--yaw_max", type=float, default=1.0)
    parser.add_argument("--vx_step", type=float, default=0.1)
    parser.add_argument("--vy_step", type=float, default=0.1)
    parser.add_argument("--yaw_step", type=float, default=0.2)
    args = parser.parse_args()

    rclpy.init()
    node = SimpleRemote(args)
    try:
        rclpy.spin(node)
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
