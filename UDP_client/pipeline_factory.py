
import sys
import os
import cv2
from pathlib import Path

from video_recorder import VideoRecorder
import live_view

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
# "branch" = tree branch segmentation pipeline -> best_candidate
# "aruco"  = direct ArUco marker detection -> ibvs/sources/ArucoSource.py
DETECTION_MODE = "branch"
# "auto" tries every predefined ArUco dictionary EVERY frame (no warmup, see
# ArucoSource.py) and uses whichever finds the tag -- costs more per frame than
# pinning one. Set to a specific name (e.g. "DICT_4X4_50") once you know your
# tag's dictionary, to keep detection cheap on every frame.
#ARUCO_DICTIONARY = "DICT_ARUCO_ORIGINAL"
ARUCO_DICTIONARY = "DICT_5X5_1000"  # superset of 5X5_50/100/250 (same first IDs), so it detects all four


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


def build_pipeline(record_prefix=None):
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
    from trackers.BranchWidthTracker import BranchWidthTracker
    from trackers.TargetModel import TargetModel
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
            # Hailo AI HAT for every camera/source type (incl. mp4 playback),
            # so offline tests run the same .hef model as flight.
            from detectors.HailoSegDetector import HailoSegDetector
            dp_detector = HailoSegDetector(hef_path=str(_here / "model" / "yolov8_segmentation.hef"), conf=0.5)
            print("Detector: Hailo (.hef)")

            from trackers.ByteTrack import ByteTrack
            from postprocessing.PostProcessor import PostProcessor
            from postprocessing.masks.MaskExtraction import MaskExtraction
            from postprocessing.geometry.DistanceHeatmap import DistanceHeatmap
            from postprocessing.geometry.BitmaskSkeleton import BitmaskSkeleton
            from postprocessing.scoring.CandidateScoring import CandidateScoring
            from postprocessing.scoring.CandidateVisualizer import CandidateVisualizer
            from pipeline.DetectionPipeline import DetectionPipeline

            dp_tracker = ByteTrack(dp_detector)
            dp_postprocessor = PostProcessor([
                MaskExtraction(), DistanceHeatmap(), BitmaskSkeleton(),
                CandidateScoring(), CandidateVisualizer(),
            ])
            dp_pipeline = DetectionPipeline(dp_source, dp_detector, dp_tracker, dp_postprocessor)

            def detection_iterator_gen():
                detection_rec = None
                try:
                    for ctx in dp_pipeline.run():
                        display = ctx.debug.get("branch_score_image", ctx.frame)

                        if record_prefix is not None:
                            if detection_rec is None:
                                detection_video_path = f"{record_prefix}_detection_overlay.mp4"
                                detection_rec = VideoRecorder(detection_video_path)
                                print(f"Recording detection overlay to {detection_video_path}")
                            detection_rec.add(display)

                        if HAS_DISPLAY:
                            cv2.imshow("Detection Pipeline", display)
                            cv2.waitKey(1)
                        bc = ctx.best_candidate
                        # the best candidate's own branch mask, to seed the width measurement
                        bc_mask = next((m for _, m, _, _, tid, _ in (ctx.tracks or [])
                                        if bc is not None and tid == bc["track_id"]), None)
                        live = live_view.current()
                        chosen = False
                        if live is not None:   # the branch chosen in the live view, if any
                            bc, bc_mask, chosen = live.choose(ctx, bc, bc_mask)
                        yield {
                            "frame": ctx.frame,
                            "best_candidate": bc,
                            "mask": bc_mask,
                            "chosen": chosen,   # chosen by the operator: the lock tolerates more motion
                            # every segmented branch: confirms that a re-found target is on one
                            "branch_mask": ctx.binary_mask,
                            # grab time from the camera (DSJSource), for the packet's "age"
                            "t_frame": (ctx.source_metadata or {}).get("t_frame"),
                        }
                finally:
                    if detection_rec is not None:
                        detection_rec.close()
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

        # DetectionPipelineSource.release() is a no-op, so the callers' source.release()
        # never closed the camera or the Hailo device -- the process then hung on exit
        # with the AI HAT still open. Release both here.
        def _release():
            # Finish the detection-overlay video first: left to garbage collection at exit, it was
            # never finalized when the exit was cut short (2026-10-11 18:36 run)
            detection_iterator.close()
            dp_source.release()
            dp_detector.release()
        source.release = _release
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
        # No KLT ORB recovery in either mode. ArUco: it re-locked onto the ceiling beam next to the
        # tag (2026-10-09); a lost tag is re-locked on its next detection instead. Branch: it found
        # the stick again at a different spot (2026-10-12); the TargetModel finds the stored spot.
        recovery=config.get("recovery.enabled", True) and DETECTION_MODE not in ("aruco", "branch"),
        # Branch mode: a UAV holding still over the branch looked like a stale lock (4 healthy
        # locks dropped on 2026-10-12); the TargetModel check catches a lock on the wrong thing.
        drop_stalled=DETECTION_MODE != "branch",
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
        recovery=config.get_section("recovery"),
        # ArUco detections are ID-checked: lock on the first one, re-lock whenever the
        # tracked point is off the tag
        trust_detections=DETECTION_MODE == "aruco",
        # Branch mode: the sent size is the branch width measured from the image edges on every
        # frame (the detector doesn't run while tracking), seeded by the branch mask at lock
        size_tracker=BranchWidthTracker() if DETECTION_MODE == "branch" else None,
        # Branch mode: the first locked point is stored and only ever found again at the same spot
        # on the branch; a new point is chosen only after the operator's reset ('r')
        target_model=TargetModel() if DETECTION_MODE == "branch" else None,
    )

    return source, pipeline
