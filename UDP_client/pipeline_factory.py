
import sys
import os
import cv2
from pathlib import Path

HAS_DISPLAY = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

_UDP_CLIENT_DIR = os.path.dirname(os.path.abspath(__file__))
_IBVS_PATH = str(Path(__file__).parent.parent / "ibvs")
_DETECTION_PIPELINE_PATH = str(Path(__file__).parent.parent / "detection_pipeline")

if _IBVS_PATH not in sys.path:
    sys.path.insert(0, _IBVS_PATH)

# --- Camera source selection ---
# "dsj"    = DSJ-3079-HE USB camera (default; standard UVC, see DSJSource.py)
# "nicla"  = Nicla Vision camera over USB serial
# "camera" = Raspberry Pi camera via picamera2
# "mp4"    = a video file, for offline testing
SOURCE_TYPE = "dsj"
MP4_PATH = None  # only used when SOURCE_TYPE == "mp4"; None = detection_pipeline/sources/example.mp4

# --- Detection mode selection ---
# "branch" = tree branch segmentation pipeline -> final_point
# "aruco"  = direct ArUco marker detection -> ibvs/sources/ArucoSource.py
DETECTION_MODE = "branch"
# "auto" tries every predefined ArUco dictionary EVERY frame (no warmup, see
# ArucoSource.py) and uses whichever finds the tag -- costs more per frame than
# pinning one. Set to a specific name (e.g. "DICT_4X4_50") once you know your
# tag's dictionary, to keep detection cheap on every frame.
ARUCO_DICTIONARY = "DICT_ARUCO_ORIGINAL"


def _load_aruco_source_class():
    # Loaded by file path rather than `from sources.ArucoSource import ArucoSource` --
    # see the module docstring in ibvs/sources/ArucoSource.py for why.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "aruco_source", str(Path(_IBVS_PATH) / "sources" / "ArucoSource.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ArucoSource


def build_pipeline(record_prefix=None, fps=10):
    """record_prefix: if given (e.g. RECORDINGS_DIR / timestamp), branch mode's
    "Detection Pipeline" debug-overlay window is also written to
    f"{record_prefix}_detection_overlay.mp4" as it's produced -- it's rendered
    deep inside DetectionPipeline's postprocessing, so this is the only place
    that has the frame to record it. The main "IBVS" overlay window is built
    by the caller instead (see main_record.py), which already has everything
    needed to draw and record it without going through this function.
    """
    from feature_extraction.FASTHarrisExtractor import FASTHarrisExtractor
    from trackers.KLTTracker import KLTTracker
    from controller.PointController import PointController
    from pipeline.IBVSPipeline import IBVSPipeline
    from config import Config

    config = Config()

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
    detection_iterator = None
    try:
        _here = Path(_DETECTION_PIPELINE_PATH)
        if SOURCE_TYPE == "dsj":
            from sources.DSJSource import DSJSource
            dp_source = DSJSource()
            print("Camera: DSJ-3079-HE")
        elif SOURCE_TYPE == "nicla":
            from sources.NiclaSource import NiclaSource
            dp_source = NiclaSource()
            print("Camera: Nicla")
        elif SOURCE_TYPE == "camera":
            from sources.CameraSource import CameraSource
            dp_source = CameraSource()
            print("Camera: Raspberry Pi camera (picamera2)")
        else:
            from sources.MP4Source import MP4Source
            video_path = MP4_PATH or str(_here / "sources" / "example.mp4")
            dp_source = MP4Source(video_path)
            print(f"Camera: MP4 ({video_path})")

        if DETECTION_MODE == "branch":
            if SOURCE_TYPE in ("dsj", "nicla"):
                from detectors.HailoSegDetector import HailoSegDetector
                dp_detector = HailoSegDetector(hef_path=str(_here / "model" / "yolov8_segmentation.hef"), conf=0.4)
                print("Detector: Hailo (.hef)")
            else:
                from detectors.YOLOBranchSeg import YOLOBranchSeg
                dp_detector = YOLOBranchSeg(model_path=str(_here / "model" / "best_small.pt"), conf=0.4)
                print("Detector: YOLO (.pt)")

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

            def detection_iterator_gen():
                detection_writer = None
                try:
                    for ctx in dp_pipeline.run():
                        display = ctx.debug.get("branch_score_image", ctx.frame)

                        if record_prefix is not None:
                            if detection_writer is None:
                                dh, dw = display.shape[:2]
                                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                detection_video_path = f"{record_prefix}_detection_overlay.mp4"
                                detection_writer = cv2.VideoWriter(detection_video_path, fourcc, fps, (dw, dh))
                                print(f"Recording detection overlay to {detection_video_path}")
                            detection_writer.write(display)

                        if HAS_DISPLAY:
                            cv2.imshow("Detection Pipeline", display)
                            cv2.waitKey(1)
                        yield {
                            "frame": ctx.frame,
                            "final_point": ctx.final_point,
                            "best_candidate": ctx.best_candidate,
                            "reference_frame": ctx.reference_frame,
                        }
                finally:
                    if detection_writer is not None:
                        detection_writer.release()
            detection_iterator = detection_iterator_gen()
    finally:
        if _DETECTION_PIPELINE_PATH in sys.path:
            sys.path.remove(_DETECTION_PIPELINE_PATH)
        if _IBVS_PATH not in sys.path:
            sys.path.insert(0, _IBVS_PATH)

    if DETECTION_MODE == "aruco":
        ArucoSource = _load_aruco_source_class()
        source = ArucoSource(dp_source, dictionary=ARUCO_DICTIONARY)
        print(f"Detection mode: ArUco (dict={ARUCO_DICTIONARY})")
    else:
        from sources.DetectionPipelineSource import DetectionPipelineSource
        source = DetectionPipelineSource(detection_iterator)
        print("Detection mode: branch (segmentation)")

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
        # Branch mode only: once KLT locks, read the camera directly instead
        # of through DetectionPipelineSource, so the expensive Hailo detect +
        # postprocessing chain only runs while actually searching for a lock,
        # not on every already-tracked frame. None in ArUco mode, where
        # `source` never runs anything heavy in the first place.
        raw_source=dp_source if DETECTION_MODE == "branch" else None,
    )

    return source, pipeline
