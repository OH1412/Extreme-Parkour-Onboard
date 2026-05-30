import os
import select
import struct
import termios
import time
from dataclasses import dataclass

import yaml


NUM_JOINTS = 12
# These constants intentionally mirror this repository's edited elmap SDK:
#   /home/rc_kfs/el_ws/elmap-rl-controller/deploy_cpp/include/Unitree_Motor/unitreeMotor/include/motor_msg_GO-M8010-6.h
# The C++ implementation links the prebuilt libUnitreeMotorSDK_*.so, so the
# Python path implements the same public packet layout and motor_driver.cpp
# joint-side <-> motor-side conversion.
GO_M8010_6_FOC_MODE = 1
GO_M8010_6_BAUD = 4000000


def _clamp(value, low, high):
    return max(low, min(high, value))


def _i16(value):
    return int(_clamp(round(value), -32768, 32767))


def _i32(value):
    return int(_clamp(round(value), -2147483648, 2147483647))


def _u16(value):
    return int(_clamp(round(value), 0, 65535))


def crc_ccitt(crc, data):
    """CRC-CCITT variant copied from elmap's Unitree_Motor/crc_ccitt.h."""
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0x8408
            else:
                crc >>= 1
            crc &= 0xFFFF
    return crc & 0xFFFF


def pack_go_m8010_6_cmd(motor_id, mode, q, dq, kp, kd, tau):
    """Pack elmap GO-M8010-6 ControlData_t.

    Layout from motor_msg_GO-M8010-6.h:
      head FE EE, mode bitfield, tau q8, dq q7, q q15, kp q15, kd q15, CRC16.
    """
    mode_byte = (int(motor_id) & 0x0F) | ((int(mode) & 0x07) << 4)
    payload = struct.pack(
        "<BBBhhIHH",
        0xFE,
        0xEE,
        mode_byte,
        _i16(tau * 256.0),
        _i16(dq * 128.0),
        _i32(q * 32768.0) & 0xFFFFFFFF,
        _u16(kp * 32768.0),
        _u16(kd * 32768.0),
    )
    return payload + struct.pack("<H", crc_ccitt(0, payload))


@dataclass
class GoM80106Feedback:
    correct: bool = False
    motor_id: int = 0
    mode: int = 0
    temp: int = 0
    merror: int = 0
    q: float = 0.0
    dq: float = 0.0
    tau: float = 0.0
    foot_force: int = 0


def unpack_go_m8010_6_feedback(packet, expected_motor_id=None):
    """Unpack elmap GO-M8010-6 MotorData_t feedback packet."""
    if len(packet) != 16 or packet[0] != 0xFE or packet[1] != 0xEE:
        return GoM80106Feedback()
    if crc_ccitt(0, packet[:14]) != struct.unpack_from("<H", packet, 14)[0]:
        return GoM80106Feedback()

    mode_byte = packet[2]
    motor_id = mode_byte & 0x0F
    mode = (mode_byte >> 4) & 0x07
    torque_raw, speed_raw, pos_raw, temp, status = struct.unpack_from("<hhibH", packet, 3)
    if expected_motor_id is not None and motor_id != int(expected_motor_id):
        return GoM80106Feedback()
    return GoM80106Feedback(
        correct=True,
        motor_id=motor_id,
        mode=mode,
        temp=temp,
        merror=status & 0x07,
        q=float(pos_raw) / 32768.0,
        dq=float(speed_raw) / 128.0,
        tau=float(torque_raw) / 256.0,
        foot_force=(status >> 3) & 0x0FFF,
    )


class RawSerialPort:
    BAUD_RATES = {
        9600: termios.B9600,
        115200: termios.B115200,
        921600: termios.B921600,
        4000000: termios.B4000000,
    }

    def __init__(self, port, baudrate=GO_M8010_6_BAUD, timeout=0.02):
        if baudrate not in self.BAUD_RATES:
            raise ValueError(f"Unsupported baudrate {baudrate}")
        self.port = port
        self.timeout = float(timeout)
        self.fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        self._configure(baudrate)
        self._rx = bytearray()

    def close(self):
        try:
            os.close(self.fd)
        except OSError:
            pass

    def send_recv(self, frame, recv_len=16):
        os.write(self.fd, frame)
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            readable, _, _ = select.select([self.fd], [], [], max(0.0, deadline - time.monotonic()))
            if not readable:
                continue
            try:
                chunk = os.read(self.fd, 256)
            except BlockingIOError:
                continue
            if not chunk:
                continue
            self._rx.extend(chunk)
            packet = self._extract_packet(recv_len)
            if packet is not None:
                return packet
        return bytes()

    def _extract_packet(self, recv_len):
        while len(self._rx) >= recv_len:
            head = self._rx.find(b"\xFE\xEE")
            if head < 0:
                del self._rx[:-1]
                return None
            if head > 0:
                del self._rx[:head]
            if len(self._rx) < recv_len:
                return None
            packet = bytes(self._rx[:recv_len])
            del self._rx[:recv_len]
            return packet
        return None

    def _configure(self, baudrate):
        attrs = termios.tcgetattr(self.fd)
        attrs[0] = 0
        attrs[1] = 0
        attrs[2] = termios.CS8 | termios.CLOCAL | termios.CREAD
        attrs[3] = 0
        attrs[4] = self.BAUD_RATES[baudrate]
        attrs[5] = self.BAUD_RATES[baudrate]
        attrs[6][termios.VMIN] = 0
        attrs[6][termios.VTIME] = 0
        termios.tcsetattr(self.fd, termios.TCSANOW, attrs)


class PythonUnitreeMotorDriver:
    """Python GO-M8010-6 motor driver matching elmap MotorDriver flow."""

    def __init__(self, config_file, port0=None, port1=None, baudrate=GO_M8010_6_BAUD, timeout=0.02):
        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)

        self.port0 = port0 or self.config.get("port0", "/dev/ttyUSB0")
        self.port1 = port1 or self.config.get("port1", "/dev/ttyUSB1")
        self.serials = [
            RawSerialPort(self.port0, baudrate=baudrate, timeout=timeout),
            RawSerialPort(self.port1, baudrate=baudrate, timeout=timeout),
        ]

        self.joint_names = list(self.config["joint_names"])
        self.default_dof_pos = [float(x) for x in self.config["default_dof_pos"]]
        self.ratio = [float(x) for x in self.config["joint_transmission_ratio"]]
        self.motor_ids = [int(x) for x in self.config["joint_mapping"]]
        self.is_reversed = [bool(x) for x in self.config["motor_is_reversed"]]
        self.port_idx = [0 if motor_id <= 6 else 1 for motor_id in self.motor_ids]

        self.motor_offsets = [0.0] * NUM_JOINTS
        self.dof_pos = list(self.default_dof_pos)
        self.dof_vel = [0.0] * NUM_JOINTS
        self.dof_tau = [0.0] * NUM_JOINTS
        self.motor_temps = [0.0] * NUM_JOINTS
        self.motor_errors = [0] * NUM_JOINTS
        self.calibrate_offsets()

    def close(self):
        for serial in self.serials:
            serial.close()

    def send_commands(self, target_dof_pos, kp, kd):
        for i in range(NUM_JOINTS):
            self.send_single(i, float(target_dof_pos[i]), 0.0, float(kp[i]), float(kd[i]), 0.0)

    def send_damping(self, kd):
        for i in range(NUM_JOINTS):
            self.send_single(i, 0.0, 0.0, 0.0, float(kd), 0.0)

    def set_zero_torque(self):
        for i in range(NUM_JOINTS):
            self.send_single(i, 0.0, 0.0, 0.0, 0.0, 0.0)

    def calibrate_offsets(self):
        for _ in range(2):
            for i in range(NUM_JOINTS):
                self._query_zero_torque(i)
        for i in range(NUM_JOINTS):
            fbk = self._query_zero_torque(i)
            if not fbk.correct:
                continue
            direction = -1.0 if self.is_reversed[i] else 1.0
            self.motor_offsets[i] = fbk.q - direction * self.default_dof_pos[i] * self.ratio[i]
            self.dof_pos[i] = self.default_dof_pos[i]
            self.dof_vel[i] = direction * fbk.dq / self.ratio[i]
            self.dof_tau[i] = direction * fbk.tau * self.ratio[i]

    def _query_zero_torque(self, dof_idx):
        return self._send_motor_frame(dof_idx, 0.0, 0.0, 0.0, 0.0, 0.0)

    def send_single(self, dof_idx, q_joint, dq_joint, kp, kd, tau):
        direction = -1.0 if self.is_reversed[dof_idx] else 1.0
        ratio = self.ratio[dof_idx]
        q_motor = direction * q_joint * ratio + self.motor_offsets[dof_idx]
        dq_motor = direction * dq_joint * ratio
        tau_motor = direction * tau / ratio
        kp_motor = kp / (ratio * ratio)
        kd_motor = kd / (ratio * ratio)
        return self._send_motor_frame(dof_idx, q_motor, dq_motor, kp_motor, kd_motor, tau_motor)

    def _send_motor_frame(self, dof_idx, q_motor, dq_motor, kp_motor, kd_motor, tau_motor):
        motor_id = self.motor_ids[dof_idx]
        frame = pack_go_m8010_6_cmd(
            motor_id,
            GO_M8010_6_FOC_MODE,
            q_motor,
            dq_motor,
            kp_motor,
            kd_motor,
            tau_motor,
        )
        packet = self.serials[self.port_idx[dof_idx]].send_recv(frame, recv_len=16)
        fbk = unpack_go_m8010_6_feedback(packet, expected_motor_id=motor_id) if packet else GoM80106Feedback()
        if fbk.correct:
            direction = -1.0 if self.is_reversed[dof_idx] else 1.0
            ratio = self.ratio[dof_idx]
            self.dof_pos[dof_idx] = direction * (fbk.q - self.motor_offsets[dof_idx]) / ratio
            self.dof_vel[dof_idx] = direction * fbk.dq / ratio
            self.dof_tau[dof_idx] = direction * fbk.tau * ratio
            self.motor_temps[dof_idx] = float(fbk.temp)
            self.motor_errors[dof_idx] = int(fbk.merror)
        return fbk
