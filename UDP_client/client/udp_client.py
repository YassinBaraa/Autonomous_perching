import socket
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class UDPSender:
    def __init__(self, host="192.168.0.128", port=5005):
        self._addr = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        logger.info(f"UDP sender -> {self._addr}")

    def send(self, error_x, error_y, distance_mm=None):
        try:
            payload = json.dumps({
                "error_x": float(error_x),
                "error_y": float(error_y),
                "tof": float(distance_mm) if distance_mm is not None else -1.0,
                "timestamp": datetime.now().timestamp(),
            }).encode()
            self._sock.sendto(payload, self._addr)
            print("packet sent \n")
        except Exception as e:
            logger.warning(f"UDP send failed: {e}")

    def close(self):
        self._sock.close()
