#!/usr/bin/env python3
import sys
import os
import cv2
import numpy as np

HAS_DISPLAY = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
from pathlib import Path
from datetime import datetime

_UDP_CLIENT_DIR = os.path.dirname(os.path.abspath(__file__))
_IBVS_PATH = str(Path(__file__).parent.parent / "ibvs")
_DETECTION_PIPELINE_PATH = str(Path(__file__).parent.parent / "detection_pipeline")

sys.path.insert(0, _UDP_CLIENT_DIR)
if _IBVS_PATH not in sys.path:
    sys.path.insert(0, _IBVS_PATH)

from client.udp_client import UDPSender

RECORDINGS_DIR = Path(__file__).parent / "recordings"
FPS = 10

# --- Source selection ---
# Set USE_NICLA = True for Nicla camera + Hailo (.hef)
# Set USE_NICLA = False for MP4 + YOLO (.pt)
#USE_NICLA = False
USE_NICLA = True


def main():
    from sources.DetectionPipelineSource import DetectionPipelineSource
    from feature_extraction.FASTHarrisExtractor import FASTHarrisExtractor
    from trackers.KLTTracker import KLTTracker
    from controller.PointController import PointController
    from pipeline.IBVSPipeline import IBVSPipeline
    from config import Config

    config = Config()

    # Build detection pipeline iterator with the chosen source
    if _IBVS_PATH in sys.path:
        sys.path.remove(_IBVS_PATH)
    if _DETECTION_PIPELINE_PATH not in sys.path:
        sys.path.insert(0, _DETECTION_PIPELINE_PATH)
    for mod in list(sys.modules.keys()):
        if any(
            mod == n or mod.startswith(n + ".")
            for n in ("pipeline", "sources", "config", "postprocessing", "main",
                      "detectors", "trackers", "feature_extraction", "controller")
        ):
            del sys.modules[mod]
    try:
        _here = Path(_DETECTION_PIPELINE_PATH)
        if USE_NICLA:
            from sources.NiclaSource import NiclaSource
            from detectors.HailoSegDetector import HailoSegDetector
            dp_source = NiclaSource()
            dp_detector = HailoSegDetector(hef_path=str(_here / "model" / "yolov8_segmentation.hef"), conf=0.4)
            print("Source: Nicla + Hailo (.hef)")
        else:
            from sources.MP4Source import MP4Source
            from detectors.YOLOBranchSeg import YOLOBranchSeg
            dp_source = MP4Source(str(_here / "sources" / "example.mp4"))
            dp_detector = YOLOBranchSeg(model_path=str(_here / "model" / "best_small.pt"), conf=0.4)
            print("Source: MP4 + YOLO (.pt)")

        from trackers.ByteTrack import ByteTrack
        from postprocessing.PostProcessor import PostProcessor
        from postprocessing.masks.MaskExtraction import MaskExtraction
        from postprocessing.geometry.DistanceHeatmap import DistanceHeatmap
        from postprocessing.geometry.BitmaskSkeleton import BitmaskSkeleton
        from postprocessing.scoring.CandidateScoring import CandidateScoring
        from postprocessing.scoring.WarmupFinalPoint import WarmupFinalPoint
        from postprocessing.scoring.CandidateVisualizer import CandidateVisualizer
        from pipeline.DetectionPipeline import DetectionPipeline

        dp_tracker = ByteTrack(dp_detector)
        dp_postprocessor = PostProcessor([
            MaskExtraction(), DistanceHeatmap(), BitmaskSkeleton(),
            CandidateScoring(), WarmupFinalPoint(), CandidateVisualizer(),
        ])
        dp_pipeline = DetectionPipeline(dp_source, dp_detector, dp_tracker, dp_postprocessor)
        def detection_iterator():
            for ctx in dp_pipeline.run():
                display = ctx.debug.get("branch_score_image", ctx.frame)
                if HAS_DISPLAY:
                    cv2.imshow("Detection Pipeline", display)
                    cv2.waitKey(1)
                yield {
                    "frame": ctx.frame,
                    "final_point": ctx.final_point,
                    "best_candidate": ctx.best_candidate,
                    "reference_frame": ctx.reference_frame,
                    "distance_mm": (ctx.source_metadata or {}).get("distance_mm"),
                    "frame_w": ctx.frame.shape[1],
                    "frame_h": ctx.frame.shape[0],
                }
        detection_iterator = detection_iterator()
    finally:
        if _DETECTION_PIPELINE_PATH in sys.path:
            sys.path.remove(_DETECTION_PIPELINE_PATH)
        if _IBVS_PATH not in sys.path:
            sys.path.insert(0, _IBVS_PATH)

    source = DetectionPipelineSource(detection_iterator)

    feature_extractor = FASTHarrisExtractor(
        max_features=config.get("feature_extraction.max_features"),
        fast_threshold=config.get("feature_extraction.fast_threshold"),
        harris_block_size=config.get("feature_extraction.harris_block_size"),
        harris_ksize=config.get("feature_extraction.harris_ksize"),
        harris_k=config.get("feature_extraction.harris_k"),
        point_focus_radius=config.get("feature_extraction.point_focus_radius"),
    )

    tracker = KLTTracker(
        feature_extractor=feature_extractor,
        min_features=config.get("controller.min_features", 8),
    )

    controller = PointController(
        gain=config.get("controller.main_gain", 0.5),
    )

    pipeline = IBVSPipeline(
        source=source,
        feature_extractor=feature_extractor,
        tracker=tracker,
        controller=controller,
    )

    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RECORDINGS_DIR / f"{timestamp}_raw.mp4"

    sender = UDPSender()
    writer = None
    frame_count = 0

    try:
        for frame_count, ctx in enumerate(pipeline.run(), 1):
            if writer is None:
                h, w = ctx.frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(str(out_path), fourcc, FPS, (w, h))
                print(f"Recording to {out_path}")

            writer.write(ctx.frame)

            ctrl = ctx.debug.get("controller", {})
            velocity = ctx.debug.get("velocity_command")
            error = ctx.debug.get("control_error_px")
            n_tracked = len(ctx.extracted_features) if ctx.extracted_features is not None else 0

            if error is not None:
                sender.send(float(error[0]), float(error[1]), ctx.distance_mm)
                print(f"[main] Frame {frame_count}: UDP sent — "
                      f"error=({error[0]:.2f}, {error[1]:.2f}), dist={ctx.distance_mm} mm")
            else:
                print(f"[main] Frame {frame_count}: no error — "
                      f"source={ctrl.get('point_source','none')}, "
                      f"tracked={n_tracked}, dist={ctx.distance_mm} mm, "
                      f"status={ctrl.get('status','?')}")

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

            if HAS_DISPLAY:
                cv2.imshow('IBVS', vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        if writer is not None:
            writer.release()
            print(f"Saved {frame_count} frames to {out_path}")
        sender.close()
        source.release()
        if HAS_DISPLAY:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
