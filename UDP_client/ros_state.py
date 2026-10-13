"""The perch controller's state, read from its ROS topic without ROS on this side.

ibvs_controller (ibvs_perching, in the uav_ros_stack Docker container on host networking) publishes
its state latched on <ns>/ibvs/state (std_msgs/String): ALIGN, ALIGNED, CLIMBING, STRIKE, PUSH,
PERCHED, TAG_LOST, ... There is no ROS outside the container, so this follows the topic directly:
XML-RPC to the ROS master for the publisher, then one TCPROS connection. A background thread keeps
retrying, so the vision runs exactly as before while ROS is not up.
"""
import os
import socket
import struct
import threading
import time
import xmlrpc.client

CALLER_ID = "/ibvs_vision_state"
STRING_MD5 = "992ce8a1687cec8c8bd883ec73ca41d1"   # std_msgs/String
RETRY_S = 2.0


def default_topic():
    ns = os.environ.get("UAV_NAMESPACE", "red").strip("/")
    return os.environ.get("IBVS_STATE_TOPIC", f"/{ns}/ibvs/state")


class _TimeoutTransport(xmlrpc.client.Transport):
    def make_connection(self, host):
        conn = super().make_connection(host)
        conn.timeout = 2.0
        return conn


class ControllerState:
    """state: the controller's latest state string, or None (no ROS / no publisher yet)."""

    def __init__(self, topic=None, master=None):
        self.topic = topic or default_topic()
        self.master = master or os.environ.get("ROS_MASTER_URI", "http://127.0.0.1:11311")
        self.state = None
        self._thread = threading.Thread(target=self._run, name="ControllerState", daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            try:
                self._follow()
            except Exception:
                pass
            self.state = None   # connection gone: unknown again
            time.sleep(RETRY_S)

    def _follow(self):
        master = xmlrpc.client.ServerProxy(self.master, transport=_TimeoutTransport())
        _, _, system = master.getSystemState(CALLER_ID)
        pubs = next((nodes for topic, nodes in system[0] if topic == self.topic), [])
        if not pubs:
            return
        _, _, node_uri = master.lookupNode(CALLER_ID, pubs[0])
        node = xmlrpc.client.ServerProxy(node_uri, transport=_TimeoutTransport())
        code, _, proto = node.requestTopic(CALLER_ID, self.topic, [["TCPROS"]])
        if code != 1:
            return
        _, host, port = proto
        with socket.create_connection((host, port), timeout=5.0) as s:
            fields = [f"callerid={CALLER_ID}", f"topic={self.topic}", f"md5sum={STRING_MD5}",
                      "type=std_msgs/String", "tcp_nodelay=1"]
            body = b"".join(struct.pack("<I", len(f)) + f.encode() for f in fields)
            s.sendall(struct.pack("<I", len(body)) + body)
            header = _read(s, struct.unpack("<I", _read(s, 4))[0])
            if b"error=" in header:
                return
            s.settimeout(None)   # the state is latched and changes rarely: wait for it
            while True:
                msg = _read(s, struct.unpack("<I", _read(s, 4))[0])
                n = struct.unpack("<I", msg[:4])[0]
                self.state = msg[4:4 + n].decode(errors="replace")


def _read(s, n):
    buf = b""
    while len(buf) < n:
        chunk = s.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("publisher closed the connection")
        buf += chunk
    return buf
