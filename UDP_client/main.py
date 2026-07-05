#!/usr/bin/env python3
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ibvs_path = str(Path(__file__).parent.parent / "ibvs")
if ibvs_path not in sys.path:
    sys.path.insert(0, ibvs_path)

from client.udp_client import UDPSender
import main as ibvs_main


def main():
    sender = UDPSender()
    try:
        for data in ibvs_main.main():
            if data["error_x"] is not None:
                sender.send(data["error_x"], data["error_y"], data["distance_mm"])
    finally:
        sender.close()


if __name__ == "__main__":
    main()
