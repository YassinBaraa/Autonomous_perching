import socket
import json
import logging

logger = logging.getLogger(__name__)


class UDPSender:
    def __init__(self, host="192.168.1.194", port=5005):
        self._addr = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        logger.info(f"UDP sender -> {self._addr}")

    def send(self, px, py):
        try:
            payload = json.dumps({
                "px": int(round(px)),
                "py": int(round(py)),
            }).encode()
            self._sock.sendto(payload, self._addr)
            print("packet sent \n")
        except Exception as e:
            logger.warning(f"UDP send failed: {e}")

    def close(self):
        self._sock.close()
