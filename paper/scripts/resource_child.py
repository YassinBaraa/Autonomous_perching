#!/usr/bin/env python3
"""Workload for resource_run.py.
  replay VIDEO OUT    the full branch pipeline (pipeline_factory, mp4 source, real Hailo detector)
                      over a recorded flight; writes (wall, frame, tracking 0/1) per frame to OUT
  detector VIDEO SEC OUT  detector + post-processing on every frame for SEC seconds"""
import csv
import io
import sys
import time
from contextlib import redirect_stdout

ROOT = "/home/crow-black/Autonomous_perching"


def replay(video, out):
    sys.path.insert(0, f"{ROOT}/UDP_client")
    import pipeline_factory as pf
    pf.SOURCE_TYPE, pf.MP4_PATH, pf.DETECTION_MODE, pf.HAS_DISPLAY = "mp4", video, "branch", False
    source, pipeline = pf.build_pipeline()
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["wall", "frame", "tracking"])
        try:
            with redirect_stdout(io.StringIO()) as buf:
                for i, ctx in enumerate(pipeline.run(), 1):
                    w.writerow([f"{time.time():.3f}", i, int(pipeline.tracker.state == "tracking")])
                    buf.seek(0); buf.truncate()
        finally:
            source.release()


def detector(video, seconds, out):
    import cv2
    sys.path.insert(0, f"{ROOT}/detection_pipeline")
    from detectors.HailoSegDetector import HailoSegDetector
    from pipeline.PipelineContext import PipelineContext
    from postprocessing.masks.MaskExtraction import MaskExtraction
    from postprocessing.geometry.DistanceHeatmap import DistanceHeatmap
    from postprocessing.geometry.BitmaskSkeleton import BitmaskSkeleton
    from postprocessing.scoring.CandidateScoring import CandidateScoring
    from postprocessing.scoring.CandidateVisualizer import CandidateVisualizer
    det = HailoSegDetector(hef_path=f"{ROOT}/detection_pipeline/model/yolov8_segmentation.hef", conf=0.5)
    stages = [MaskExtraction(), DistanceHeatmap(), BitmaskSkeleton(), CandidateScoring(), CandidateVisualizer()]
    cap = cv2.VideoCapture(video)
    t_end = time.time() + float(seconds)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["wall", "frame", "tracking"])
        i = 0
        try:
            while time.time() < t_end:
                ok, frame = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                i += 1
                _, boxes, masks, conf, ids = det.predict(frame)
                ctx = PipelineContext(frame=frame)
                ctx.masks = masks
                ctx.tracks = [(boxes[k], masks[k], conf[k], 0, k + 1, {}) for k in range(len(boxes))] if masks is not None else []
                for st in stages:
                    ctx = st.run(ctx)
                w.writerow([f"{time.time():.3f}", i, 0])
        finally:
            det.release()


if __name__ == "__main__":
    if sys.argv[1] == "replay":
        replay(sys.argv[2], sys.argv[3])
    else:
        detector(sys.argv[2], sys.argv[3], sys.argv[4])
