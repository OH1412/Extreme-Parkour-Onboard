import os
import struct
import termios
import threading
import time
from typing import Dict, Optional


class SerialTeleopController:
    """Serial teleop receiver for velocity/mode commands.

    Supported text formats per line:
      - "vx vy yaw"
      - "mode vx vy yaw"
      - "mode vx vy yaw estop"
      - "mode=2 vx=0.5 vy=0 yaw=0 estop=0"

    Supported binary format is the same payload as UdpTeleopController:
      struct "<ifffB" => mode(int32), vx(float32), vy(float32), yaw(float32), e_stop(uint8)
    """

    PACKET = struct.Struct("<ifffB")

    BAUD_RATES = {
        9600: termios.B9600,
        19200: termios.B19200,
        38400: termios.B38400,
        57600: termios.B57600,
        115200: termios.B115200,
        230400: termios.B230400,
        460800: termios.B460800,
        921600: termios.B921600,
    }

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        protocol: str = "text",
        default_mode: int = 2,
        stale_timeout: Optional[float] = 0.5,
    ):
        if protocol not in ("text", "binary"):
            raise ValueError("serial protocol must be 'text' or 'binary'")
        if baudrate not in self.BAUD_RATES:
            raise ValueError(f"Unsupported baudrate {baudrate}; add it to BAUD_RATES if needed.")

        self.port = port
        self.baudrate = baudrate
        self.protocol = protocol
        self.default_mode = int(default_mode)
        self.stale_timeout = stale_timeout
        self._latest = {
            "mode": self.default_mode,
            "vx": 0.0,
            "vy": 0.0,
            "yaw": 0.0,
            "e_stop": False,
        }
        self._last_rx_time = 0.0
        self._has_data = False
        self._running = True
        self._lock = threading.Lock()

        self._fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        self._configure_port()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def close(self):
        self._running = False
        try:
            os.close(self._fd)
        except OSError:
            pass

    def has_data(self) -> bool:
        return self._has_data

    def is_stale(self) -> bool:
        return (
            self.stale_timeout is not None
            and self._has_data
            and (time.monotonic() - self._last_rx_time) > self.stale_timeout
        )

    def get_latest(self) -> Dict[str, float]:
        with self._lock:
            cmd = dict(self._latest)
        if self.is_stale():
            cmd["vx"] = 0.0
            cmd["vy"] = 0.0
            cmd["yaw"] = 0.0
        return cmd

    def _configure_port(self):
        attrs = termios.tcgetattr(self._fd)
        attrs[0] = 0
        attrs[1] = 0
        attrs[2] = termios.CS8 | termios.CLOCAL | termios.CREAD
        attrs[3] = 0
        attrs[4] = self.BAUD_RATES[self.baudrate]
        attrs[5] = self.BAUD_RATES[self.baudrate]
        attrs[6][termios.VMIN] = 0
        attrs[6][termios.VTIME] = 1
        termios.tcsetattr(self._fd, termios.TCSANOW, attrs)

    def _read_loop(self):
        buffer = bytearray()
        while self._running:
            try:
                chunk = os.read(self._fd, 256)
            except BlockingIOError:
                time.sleep(0.002)
                continue
            except OSError:
                if self._running:
                    time.sleep(0.01)
                continue

            if not chunk:
                time.sleep(0.002)
                continue

            buffer.extend(chunk)
            if self.protocol == "text":
                self._consume_text(buffer)
            else:
                self._consume_binary(buffer)

    def _consume_text(self, buffer: bytearray):
        while True:
            newline = buffer.find(b"\n")
            if newline < 0:
                if len(buffer) > 512:
                    del buffer[:-128]
                return
            raw_line = bytes(buffer[:newline]).strip()
            del buffer[: newline + 1]
            if not raw_line:
                continue
            try:
                cmd = self._parse_text_line(raw_line.decode("ascii", errors="ignore"))
            except ValueError:
                continue
            self._set_latest(cmd)

    def _consume_binary(self, buffer: bytearray):
        while len(buffer) >= self.PACKET.size:
            packet = bytes(buffer[: self.PACKET.size])
            del buffer[: self.PACKET.size]
            mode, vx, vy, yaw, e_stop = self.PACKET.unpack(packet)
            self._set_latest(
                {
                    "mode": int(mode),
                    "vx": float(vx),
                    "vy": float(vy),
                    "yaw": float(yaw),
                    "e_stop": bool(e_stop),
                }
            )

    def _parse_text_line(self, line: str) -> Dict[str, float]:
        line = line.replace(",", " ").strip()
        if not line:
            raise ValueError("empty serial command")

        if "=" in line:
            cmd = dict(self._latest)
            for token in line.split():
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                key = key.lower()
                if key in ("mode", "m"):
                    cmd["mode"] = int(float(value))
                elif key in ("vx", "x"):
                    cmd["vx"] = float(value)
                elif key in ("vy", "y"):
                    cmd["vy"] = float(value)
                elif key in ("yaw", "w", "omega"):
                    cmd["yaw"] = float(value)
                elif key in ("estop", "e_stop", "stop"):
                    cmd["e_stop"] = bool(int(float(value)))
            return cmd

        parts = line.split()
        if len(parts) == 1 and parts[0].lower() in ("stop", "estop", "e_stop"):
            return {"mode": 0, "vx": 0.0, "vy": 0.0, "yaw": 0.0, "e_stop": True}

        values = [float(x) for x in parts]
        if len(values) == 3:
            mode = self.default_mode
            vx, vy, yaw = values
            e_stop = False
        elif len(values) == 4:
            mode, vx, vy, yaw = values
            e_stop = False
        elif len(values) >= 5:
            mode, vx, vy, yaw, e_stop = values[:5]
        else:
            raise ValueError("serial command needs 3, 4, or 5 values")

        return {
            "mode": int(mode),
            "vx": float(vx),
            "vy": float(vy),
            "yaw": float(yaw),
            "e_stop": bool(int(e_stop)),
        }

    def _set_latest(self, cmd: Dict[str, float]):
        with self._lock:
            self._latest = cmd
            self._last_rx_time = time.monotonic()
            self._has_data = True
