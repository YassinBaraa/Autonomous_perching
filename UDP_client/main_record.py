#!/usr/bin/env python3
import sys
import os
import cv2
import numpy as np
import time
from pathlib import Path
from datetime import datetime

_UDP_CLIENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _UDP_CLIENT_DIR)

from pipeline_factory import build_pipeline, HAS_DISPLAY
from client.udp_client import UDPSender
from video_recorder import VideoRecorder
import reset_key

RECORDINGS_DIR = Path(__file__).parent / "recordings"


def main():
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    record_prefix = RECORDINGS_DIR / timestamp
    out_path = f"{record_prefix}_raw.mp4"
    ibvs_out_path = f"{record_prefix}_ibvs_overlay.mp4"

    source, pipeline = build_pipeline(record_prefix=record_prefix)

    sender = UDPSender()
    reset_key.start()
    # Encoded on background threads (one per file), so the loop never waits for the encoder
    raw_rec = VideoRecorder(out_path)
    ibvs_rec = VideoRecorder(ibvs_out_path)
    print(f"Recording raw to {out_path}")
    print(f"Recording IBVS overlay to {ibvs_out_path}")
    frame_count = 0

    try:
        t_prev_frame = time.monotonic()
        for frame_count, ctx in enumerate(pipeline.run(), 1):
            reset_key.check(pipeline, ctx)
            t_loop0 = time.monotonic()

            t_w1_0 = time.monotonic()
            raw_rec.add(ctx.frame)
            t_w1_1 = time.monotonic()

            ctrl = ctx.debug.get("controller", {})
            velocity = ctx.debug.get("velocity_command")
            n_tracked = len(ctx.extracted_features) if ctx.extracted_features is not None else 0
            # Only a locked, tracked point is ever sent (nothing while idle or lost)
            target_point = ctx.estimated_point

            t_udp0 = time.monotonic()
            if target_point is not None:
                sender.send(int(round(target_point[0])), int(round(target_point[1])),
                            size=ctx.target_size, t_frame=ctx.t_frame)
                print(f"[main] Frame {frame_count}: UDP sent — "
                      f"point=({target_point[0]}, {target_point[1]})")
            else:
                print(f"[main] Frame {frame_count}: no target point — "
                      f"state=no lock/lost, tracked={n_tracked}")
            t_udp1 = time.monotonic()

            vis = ctx.frame.copy()
            h, w = vis.shape[:2]
            center = (w // 2, h // 2)

            if ctx.extracted_features is not None:
                for (x, y) in ctx.extracted_features:
                    cv2.circle(vis, (int(x), int(y)), 4, (0, 255, 0), -1)

            if ctx.point is not None:
                cv2.circle(vis, (int(ctx.point[0]), int(ctx.point[1])), 7, (255, 0, 0), 2)
            if ctx.estimated_point is not None:
                cv2.circle(vis, (int(ctx.estimated_point[0]), int(ctx.estimated_point[1])), 7, (0, 0, 255), 2)

            cv2.drawMarker(vis, center, (200, 200, 200), cv2.MARKER_CROSS, 20, 1)

            if velocity is not None and np.linalg.norm(velocity) > 0.5:
                tip = (
                    int(center[0] + velocity[0] * 2.0),
                    int(center[1] + velocity[1] * 2.0),
                )
                cv2.arrowedLine(vis, center, tip, (0, 165, 255), 2, tipLength=0.3)

            t_w2_0 = time.monotonic()
            ibvs_rec.add(vis)
            t_w2_1 = time.monotonic()

            if HAS_DISPLAY:
                cv2.imshow('IBVS', vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            t_loop1 = time.monotonic()
            print(f"[main] Frame {frame_count}: TIMING — "
                  f"raw_write={1000*(t_w1_1-t_w1_0):.0f}ms "
                  f"udp_send={1000*(t_udp1-t_udp0):.0f}ms "
                  f"overlay_write={1000*(t_w2_1-t_w2_0):.0f}ms "
                  f"loop_total={1000*(t_loop1-t_loop0):.0f}ms "
                  f"since_prev_frame={1000*(t_loop0-t_prev_frame):.0f}ms")
            t_prev_frame = t_loop0

    finally:
        sender.close()
        source.release()
        # Hardware first, then wait for the encoders to finish what is queued and finalize the files
        raw_rec.close()
        ibvs_rec.close()
        if raw_rec.written:
            print(f"Saved {raw_rec.written} frames at {raw_rec.fps} fps to {out_path} and {ibvs_out_path}")
        if raw_rec.dropped or ibvs_rec.dropped:
            print(f"Encoder fell behind: {raw_rec.dropped} raw / {ibvs_rec.dropped} overlay frames "
                  f"left out of the recording (the loop itself was not slowed)")
        if HAS_DISPLAY:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
