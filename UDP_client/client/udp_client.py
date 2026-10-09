import socket
import json
import logging
import math
import time

logger = logging.getLogger(__name__)


class UDPSender:
    def __init__(self, host="192.168.0.128", port=5005):
        self._addr = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        logger.info(f"UDP sender -> {self._addr}")

    def send(self, px, py, size=None, t_frame=None):
        """size: target's apparent size [px] on this frame; t_frame: time.monotonic()
        when the frame was grabbed. Both optional: an invalid value leaves its key out
        (never 0, NaN or null -- the receiver drops a packet it cannot parse)."""
        try:
            packet = {
                "px": int(round(px)),
                "py": int(round(py)),
            }
            if size is not None and math.isfinite(size):
                size = round(float(size), 2)
                if size > 0:
                    packet["size"] = size
            if t_frame is not None:
                age = round(time.monotonic() - t_frame, 4)  # grab -> send, on this computer's clock
                if 0.0 <= age < 1.0:
                    packet["age"] = age
            payload = json.dumps(packet).encode()
            self._sock.sendto(payload, self._addr)
            print(f"packet sent {payload.decode()} \n")
        except Exception as e:
            logger.warning(f"UDP send failed: {e}")

    def close(self):
        self._sock.close()
