#!/usr/bin/env python3
import sys
import os
import cv2
import json
import numpy as np
import time
from pathlib import Path
from datetime import datetime

_UDP_CLIENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _UDP_CLIENT_DIR)

import pipeline_factory
from pipeline_factory import build_pipeline, HAS_DISPLAY
from client.udp_client import UDPSender
from video_recorder import VideoRecorder
from overlay import frame_record, code_version
from ros_state import ControllerState
import reset_key

RECORDINGS_DIR = Path(__file__).parent / "recordings"
# Controller states (ibvs_perching, <ns>/ibvs/state) in which the UAV climbs to the branch: the
# vision then holds the point -- it is only tracked, never moved by corrections or a re-pick.
HOLD_STATES = {"CLIMBING", "STRIKE", "PUSH", "PERCHED"}


def main():
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    record_prefix = RECORDINGS_DIR / timestamp
    out_path = f"{record_prefix}_raw.mp4"
    data_path = f"{record_prefix}_overlay.jsonl"

    source, pipeline = build_pipeline(record_prefix=record_prefix)

    sender = UDPSender()
    controller = ControllerState()
    print(f"Following the controller state on {controller.topic} (point held while {', '.join(sorted(HOLD_STATES))})")
    ctrl_state = None
    reset_key.start()
    # Encoded on a background thread, so the loop never waits for the encoder
    raw_rec = VideoRecorder(out_path)
    # The overlay video is not drawn during the flight (drawing live slowed the loop ~12 %): each
    # frame's values go to a small data file, and render_overlay.sh draws the video afterwards
    data_file = open(data_path, "w")
    data_file.write(json.dumps(dict(header=True, mode=pipeline_factory.DETECTION_MODE,
                                    version=code_version(Path(_UDP_CLIENT_DIR).parent))) + "\n")
    print(f"Recording raw to {out_path}")
    print(f"Recording overlay data to {data_path} (draw the video afterwards: ./render_overlay.sh)")
    frame_count = 0
    rate_hz = None

    try:
        t_prev_frame = time.monotonic()
        for frame_count, ctx in enumerate(pipeline.run(), 1):
            reset_key.check(pipeline, ctx)
            t_loop0 = time.monotonic()
            if controller.state != ctrl_state:
                ctrl_state = controller.state
                print(f"[main] Frame {frame_count}: controller state {ctrl_state or 'unknown (no ROS)'}")
            if hasattr(pipeline, "set_holding"):
                pipeline.set_holding(ctrl_state in HOLD_STATES, f"controller {ctrl_state}")

            t_w1_0 = time.monotonic()
            raw_written = raw_rec.add(ctx.frame)
            t_w1_1 = time.monotonic()

            n_tracked = len(ctx.extracted_features) if ctx.extracted_features is not None else 0
            # Only a locked, tracked point is ever sent (nothing while idle or lost)
            target_point = ctx.estimated_point

            t_udp0 = time.monotonic()
            packet = None
            if target_point is not None:
                packet = sender.send(int(round(target_point[0])), int(round(target_point[1])),
                                     size=ctx.target_size, t_frame=ctx.t_frame)
                print(f"[main] Frame {frame_count}: UDP sent — "
                      f"point=({target_point[0]}, {target_point[1]})")
            else:
                print(f"[main] Frame {frame_count}: no target point — "
                      f"state=no lock/lost, tracked={n_tracked}")
            t_udp1 = time.monotonic()

            dt = t_loop0 - t_prev_frame
            if dt > 0:
                rate_hz = 1.0 / dt if rate_hz is None else 0.9 * rate_hz + 0.1 / dt
            t_w2_0 = time.monotonic()
            data_file.write(json.dumps(dict(frame_record(ctx, pipeline, packet, rate_hz, raw_written),
                                            ctrl=ctrl_state)) + "\n")
            t_w2_1 = time.monotonic()

            if HAS_DISPLAY:   # live view: camera image with just the target point (cheap)
                vis = ctx.frame.copy()
                if target_point is not None:
                    cv2.circle(vis, (int(target_point[0]), int(target_point[1])), 12, (151, 220, 61), 2, cv2.LINE_AA)
                cv2.imshow('IBVS', vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            t_loop1 = time.monotonic()
            print(f"[main] Frame {frame_count}: TIMING — "
                  f"raw_write={1000*(t_w1_1-t_w1_0):.0f}ms "
                  f"udp_send={1000*(t_udp1-t_udp0):.0f}ms "
                  f"data_write={1000*(t_w2_1-t_w2_0):.0f}ms "
                  f"loop_total={1000*(t_loop1-t_loop0):.0f}ms "
                  f"since_prev_frame={1000*(t_loop0-t_prev_frame):.0f}ms")
            t_prev_frame = t_loop0

    finally:
        sender.close()
        source.release()
        # Hardware first, then wait for the encoders to finish what is queued and finalize the files
        raw_rec.close()
        data_file.close()
        if raw_rec.written:
            print(f"Saved {raw_rec.written} frames at {raw_rec.fps} fps to {out_path}, overlay data to {data_path}")
            print("Draw the overlay video with: ./render_overlay.sh")
        if raw_rec.dropped:
            print(f"Encoder fell behind: {raw_rec.dropped} raw frames left out of the recording "
                  f"(the loop itself was not slowed)")
        if HAS_DISPLAY:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
