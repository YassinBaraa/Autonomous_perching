"""MP4 recording off the main loop: each VideoRecorder encodes one file on its own thread.

Encoding inline made main_record.py's detection loop wait for the encoder on every
frame. Here the loop only hands a frame over and moves on.
"""
import queue
import threading
import time

import cv2

PROBE_FRAMES = 30   # frames held back to measure the real frame rate, as in detection_pipeline/record_video.py
DEFAULT_FPS = 10.0  # stamp for a run too short to measure


class VideoRecorder:
    """One MP4, written on a background thread.

    The file is opened once PROBE_FRAMES frames have arrived and is stamped with the rate
    they actually arrived at, so it plays back in real time. If the encoder falls
    max_backlog frames behind, new frames are left out of the recording (never out of the
    loop) and counted in `dropped`.
    """

    def __init__(self, path, max_backlog=PROBE_FRAMES):
        self.path = str(path)
        self.fps = None
        self.written = 0
        self.dropped = 0
        self._q = queue.Queue(maxsize=max_backlog)
        self._thread = threading.Thread(target=self._run, name=f"VideoRecorder {self.path}")
        self._thread.start()

    def add(self, frame):
        """Queue a copy of `frame`, stamped now. Never waits."""
        try:
            self._q.put_nowait((frame.copy(), time.monotonic()))
        except queue.Full:
            self.dropped += 1

    def close(self):
        """Encode what is still queued and finalize the file; returns once the file is complete."""
        while self._thread.is_alive():
            try:
                self._q.put(None, timeout=0.5)
                break
            except queue.Full:
                pass
        self._thread.join()

    def _run(self):
        probe, writer = [], None
        try:
            while True:
                try:
                    item = self._q.get(timeout=0.5)
                except queue.Empty:
                    if threading.main_thread().is_alive():
                        continue
                    break  # main thread ended without close() (e.g. a second Ctrl+C): finalize anyway
                if item is None:
                    break
                if writer is not None:
                    self._write(writer, item[0])
                    continue
                probe.append(item)
                if len(probe) == PROBE_FRAMES:
                    writer = self._open(probe)
                    probe = []
            if writer is None and probe:  # stopped before the probe filled
                writer = self._open(probe)
        finally:
            if writer is not None:
                writer.release()

    def _open(self, probe):
        """Open the file stamped with the probe frames' measured rate, and write them."""
        span = probe[-1][1] - probe[0][1]
        self.fps = round((len(probe) - 1) / span, 2) if len(probe) > 1 and span > 0 else DEFAULT_FPS
        h, w = probe[0][0].shape[:2]
        writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (w, h))
        if not writer.isOpened():
            print(f"[VideoRecorder] Could not open {self.path} for writing")
        for frame, _ in probe:
            self._write(writer, frame)
        return writer

    def _write(self, writer, frame):
        if not writer.isOpened():
            return
        try:
            writer.write(frame)
            self.written += 1
        except cv2.error as e:  # keep draining the queue, so close() can never block on it
            print(f"[VideoRecorder] {self.path}: {e}")
