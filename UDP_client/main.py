#!/usr/bin/env python3
import sys
import os
from pathlib import Path
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline_factory import build_pipeline
from client.udp_client import UDPSender
import reset_key


def main():
    source, pipeline = build_pipeline()
    sender = UDPSender()
    reset_key.start()
    try:
        start = time.time()
        for ctx in pipeline.run():
            reset_key.check(pipeline, ctx)
            

            # Only a locked, tracked point is ever sent (nothing while idle or lost)
            target_point = ctx.estimated_point

            if target_point is not None:
                sender.send(int(round(target_point[0])), int(round(target_point[1])),
                            size=ctx.target_size, t_frame=ctx.t_frame)

            end = time.time()
            print(f"Time delay to compute: {end-start} \n")
            start = time.time()

    finally:
        sender.close()
        source.release()


if __name__ == "__main__":
    main()
