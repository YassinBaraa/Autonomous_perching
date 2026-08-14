#!/usr/bin/env python3
import sys
import os
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

_UDP_CLIENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _UDP_CLIENT_DIR)

from pipeline_factory import build_pipeline, HAS_DISPLAY
from client.udp_client import UDPSender

RECORDINGS_DIR = Path(__file__).parent / "recordings"
FPS = 10


def main():
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    record_prefix = RECORDINGS_DIR / timestamp
    out_path = f"{record_prefix}_raw.mp4"
    ibvs_out_path = f"{record_prefix}_ibvs_overlay.mp4"

    source, pipeline = build_pipeline(record_prefix=record_prefix, fps=FPS)

    sender = UDPSender()
    writer = None
    ibvs_writer = None
    frame_count = 0

    try:
        for frame_count, ctx in enumerate(pipeline.run(), 1):
            if writer is None:
                h, w = ctx.frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(str(out_path), fourcc, FPS, (w, h))
                ibvs_writer = cv2.VideoWriter(str(ibvs_out_path), fourcc, FPS, (w, h))
                print(f"Recording raw to {out_path}")
                print(f"Recording IBVS overlay to {ibvs_out_path}")

            writer.write(ctx.frame)

            ctrl = ctx.debug.get("controller", {})
            velocity = ctx.debug.get("velocity_command")
            n_tracked = len(ctx.extracted_features) if ctx.extracted_features is not None else 0
            target_point = ctx.estimated_point if ctx.estimated_point is not None else ctx.point

            if target_point is not None:
                sender.send(int(round(target_point[0])), int(round(target_point[1])))
                print(f"[main] Frame {frame_count}: UDP sent — "
                      f"point=({target_point[0]}, {target_point[1]})")
            else:
                print(f"[main] Frame {frame_count}: no target point — "
                      f"source={ctrl.get('point_source','none')}, tracked={n_tracked}")

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

            ibvs_writer.write(vis)

            if HAS_DISPLAY:
                cv2.imshow('IBVS', vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        if writer is not None:
            writer.release()
            ibvs_writer.release()
            print(f"Saved {frame_count} frames to {out_path} and {ibvs_out_path}")
        sender.close()
        source.release()
        if HAS_DISPLAY:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
