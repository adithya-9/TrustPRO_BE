"""Frame extraction from interview recordings.

Browser MediaRecorder WebM files have no duration or seek index, so random seeking is
unreliable. Frames are therefore read sequentially (cheap `grab()` for skipped frames,
full decode only for frames that are kept) and timestamped with the container's
presentation time, which lines up with the gaze offsets recorded live.

Two access patterns are supported:
  * sample_frames()   - a baseline sweep at a fixed interval (default 1 s)
  * frames_at()       - specific timestamps, used to confirm anomalies with neighbouring frames
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


class FrameExtractionError(RuntimeError):
    pass


def _open(path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        capture.release()
        raise FrameExtractionError("The interview recording could not be opened.")
    return capture


def _resize(frame: np.ndarray, max_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)


def probe(path: Path) -> dict:
    capture = _open(path)
    try:
        info = {
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
            "codec": int(capture.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode("latin-1", "replace").strip("\x00"),
        }
        ok = capture.grab()
        if not ok:
            raise FrameExtractionError("The interview recording contains no readable video frames.")
        return info
    finally:
        capture.release()


def _walk(path: Path) -> Iterator[tuple[int, cv2.VideoCapture]]:
    """Yield (timestamp_ms, capture) after each successful grab()."""
    capture = _open(path)
    last_ts = -1
    index = 0
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    try:
        while capture.grab():
            ts = capture.get(cv2.CAP_PROP_POS_MSEC)
            if not np.isfinite(ts) or (ts <= 0 and index > 0):
                ts = index * 1000.0 / fps   # container without timestamps: fall back to frame counting
            index += 1
            ts_ms = int(ts)
            if ts_ms < last_ts:            # never go backwards
                ts_ms = last_ts
            last_ts = ts_ms
            yield ts_ms, capture
    finally:
        capture.release()


def sample_frames(path: Path, interval_ms: int, max_width: int = 960) -> Iterator[tuple[int, np.ndarray]]:
    next_ts = 0
    decoded = 0
    for ts_ms, capture in _walk(path):
        if ts_ms + 1 < next_ts:
            continue
        ok, frame = capture.retrieve()
        if not ok or frame is None:
            continue
        decoded += 1
        yield ts_ms, _resize(frame, max_width)
        next_ts = (ts_ms // interval_ms + 1) * interval_ms
    if decoded == 0:
        raise FrameExtractionError("No frames could be read from the interview recording.")


def frames_at(path: Path, wanted_ms: list[int], max_width: int = 960,
              tolerance_ms: int = 120) -> dict[int, tuple[int, np.ndarray]]:
    """Closest decoded frame for each requested timestamp: {wanted: (actual_ts, frame)}."""
    targets = sorted(set(max(0, t) for t in wanted_ms))
    found: dict[int, tuple[int, np.ndarray]] = {}
    if not targets:
        return found
    i = 0
    for ts_ms, capture in _walk(path):
        while i < len(targets) and ts_ms > targets[i] + tolerance_ms:
            i += 1                      # this target fell between frames we could not match
        if i >= len(targets):
            break
        if abs(ts_ms - targets[i]) <= tolerance_ms:
            ok, frame = capture.retrieve()
            if ok and frame is not None:
                resized = _resize(frame, max_width)
                while i < len(targets) and abs(ts_ms - targets[i]) <= tolerance_ms:
                    found[targets[i]] = (ts_ms, resized)
                    i += 1
    return found
