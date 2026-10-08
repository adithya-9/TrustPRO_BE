"""Environmental anomalies in the recorded assessment video.

Same logic and model as the Environment_anomalies_streamlit project (YOLO11m, Ultralytics):

* one frame per second of video, 8 frames per YOLO call (batched);
* person >= 0.50: 2 people -> "Second Person Detected", 3+ -> "Multiple Persons Detected";
* no person for 5 sampled frames in a row -> "Person Left Screen" (once, until they return);
* cell phone >= 0.35, book >= 0.35, laptop >= 0.40, tv/monitor >= 0.40, each seen in 2 sampled
  frames in a row -> "Mobile Phone / Book / Laptop / Monitor Detected";
* frame dark (mean < 30) or flat (variance < 15) for 3 frames -> "Camera Covered";
* large change from the previous frame (mean difference > 45) for 3 frames -> "Camera Position Changed";
* the same event type is logged at most once every 3 seconds.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import cv2
import numpy as np

from app.core.config import get_settings
from app.services.storage import encode_jpeg

log = logging.getLogger(__name__)

PERSON_CONFIDENCE = 0.50
MOBILE_CONFIDENCE = 0.35
BOOK_CONFIDENCE = 0.35
LAPTOP_CONFIDENCE = 0.40
MONITOR_CONFIDENCE = 0.40
OBJECT_CONFIRMATION_FRAMES = 2
MOBILE_CONFIRMATION_FRAMES = 1     # a phone is reported from a single frame (user decision, 2026-10-06)
PERSON_ABSENT_THRESHOLD = 5
DARK_THRESHOLD = 30
VARIANCE_THRESHOLD = 15
CAMERA_COVER_CONFIRMATION = 3
CAMERA_CHANGE_THRESHOLD = 45
CAMERA_CHANGE_CONFIRMATION = 3
EVENT_COOLDOWN_SECONDS = 3
BATCH_SIZE = 8

# YOLO label -> (confidence threshold, anomaly type, box colour (BGR), confirmation frames)
TRACKED_OBJECTS = {
    "cell phone": (MOBILE_CONFIDENCE, "Mobile Phone Detected", (0, 0, 255), MOBILE_CONFIRMATION_FRAMES),
    "book": (BOOK_CONFIDENCE, "Book Detected", (255, 0, 0), OBJECT_CONFIRMATION_FRAMES),
    "laptop": (LAPTOP_CONFIDENCE, "Laptop Detected", (0, 165, 255), OBJECT_CONFIRMATION_FRAMES),
    "tv": (MONITOR_CONFIDENCE, "Monitor Detected", (255, 255, 0), OBJECT_CONFIRMATION_FRAMES),
}

# The OpenVINO export has a fixed input (height, width) and takes one frame per call; see
# scripts/export_environment_model.py. The .pt model takes batches at imgsz 640.
OPENVINO_IMGSZ = (384, 640)

_model = None
_model_name = None
_model_lock = threading.Lock()
_infer_lock = threading.Lock()


def get_environment_model():
    """The YOLO model, loaded once per process (Ultralytics; first call takes a few seconds).
    Prefers the OpenVINO export (same weights, faster on CPU); falls back to the .pt file."""
    global _model, _model_name
    if _model is None:
        with _model_lock:
            if _model is None:
                from ultralytics import YOLO

                s = get_settings()
                candidates = [s.models_dir / s.environment_model, s.models_dir / s.environment_model_fallback]
                path = next((p for p in candidates if p.exists()), None)
                if path is None:
                    from app.ai.runtime import ModelUnavailableError
                    raise ModelUnavailableError(f"Environment model not found: {candidates[0].name}")
                _model = YOLO(str(path), task="detect")
                _model_name = path.name
                log.info("Loaded environment model %s", path.name)
    return _model


@dataclass
class EnvironmentEvent:
    timestamp_s: float
    type: str
    confidence: float | None
    jpeg: bytes

    def as_dict(self) -> dict:
        return {"timestamp_s": round(self.timestamp_s, 1), "type": self.type,
                "confidence": None if self.confidence is None else round(self.confidence, 4)}


class EnvironmentalDetector:
    """Per-video state of the anomaly rules (one instance per report)."""

    def __init__(self, names: dict[int, str]) -> None:
        self.names = names
        self.tracked_detected_frames = {label: 0 for label in TRACKED_OBJECTS}
        self.person_absent_frames = 0
        self.person_left_logged = False
        self.camera_covered_frames = 0
        self.camera_covered_logged = False
        self.previous_gray = None
        self.camera_change_frames = 0
        self.camera_position_logged = False

    def detect(self, frame: np.ndarray, result) -> tuple[np.ndarray, list[dict]]:
        """frame: BGR image (drawn on); result: this frame's Ultralytics result."""
        anomalies: list[dict] = []
        original = frame.copy()
        person_count = 0
        person_confidences: list[float] = []
        tracked_found: dict[str, float | None] = {label: None for label in TRACKED_OBJECTS}

        boxes = result.boxes
        for i in range(0 if boxes is None else len(boxes)):
            confidence = float(boxes.conf[i])
            label = self.names[int(boxes.cls[i])]
            x1, y1, x2, y2 = map(int, boxes.xyxy[i])
            if label == "person" and confidence >= PERSON_CONFIDENCE:
                person_count += 1
                person_confidences.append(confidence)
                _draw_box(frame, x1, y1, x2, y2, f"person{person_count}", confidence, (0, 255, 0))
                continue
            tracked = TRACKED_OBJECTS.get(label)
            if tracked and confidence >= tracked[0]:
                if tracked_found[label] is None or confidence > tracked_found[label]:
                    tracked_found[label] = confidence
                _draw_box(frame, x1, y1, x2, y2, label, confidence, tracked[2])

        # ---- people
        avg = sum(person_confidences) / len(person_confidences) if person_confidences else None
        if person_count == 2:
            anomalies.append({"type": "Second Person Detected", "confidence": avg})
        elif person_count >= 3:
            anomalies.append({"type": "Multiple Persons Detected", "confidence": avg})

        if person_count == 0:
            self.person_absent_frames += 1
        else:
            self.person_absent_frames = 0
            self.person_left_logged = False
        if self.person_absent_frames >= PERSON_ABSENT_THRESHOLD and not self.person_left_logged:
            anomalies.append({"type": "Person Left Screen", "confidence": None})
            self.person_left_logged = True

        # ---- objects, confirmed over consecutive sampled frames
        for label, (_, anomaly_type, _, confirmation) in TRACKED_OBJECTS.items():
            if tracked_found[label] is not None:
                self.tracked_detected_frames[label] += 1
            else:
                self.tracked_detected_frames[label] = 0
            if self.tracked_detected_frames[label] >= confirmation:
                anomalies.append({"type": anomaly_type, "confidence": tracked_found[label]})
                self.tracked_detected_frames[label] = 0

        # ---- camera covered
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        if np.mean(gray) < DARK_THRESHOLD or np.var(gray) < VARIANCE_THRESHOLD:
            self.camera_covered_frames += 1
        else:
            self.camera_covered_frames = 0
            self.camera_covered_logged = False
        if self.camera_covered_frames >= CAMERA_COVER_CONFIRMATION and not self.camera_covered_logged:
            anomalies.append({"type": "Camera Covered", "confidence": None})
            self.camera_covered_logged = True

        # ---- camera moved
        if self.previous_gray is not None and self.previous_gray.shape == gray.shape:
            if np.mean(cv2.absdiff(self.previous_gray, gray)) > CAMERA_CHANGE_THRESHOLD:
                self.camera_change_frames += 1
            else:
                self.camera_change_frames = 0
                self.camera_position_logged = False
            if self.camera_change_frames >= CAMERA_CHANGE_CONFIRMATION and not self.camera_position_logged:
                anomalies.append({"type": "Camera Position Changed", "confidence": None})
                self.camera_position_logged = True
        self.previous_gray = gray

        cv2.putText(frame, f"Persons: {person_count}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return frame, anomalies


def _draw_box(frame, x1, y1, x2, y2, label, confidence, color) -> None:
    h, w = frame.shape[:2]
    x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
    y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {confidence * 100:.1f}%"
    scale = max(0.6, w / 1200)
    thickness = max(1, int(scale * 2))
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    tx = max(5, min(x1, w - tw - 5))
    ty = min(max(th + base + 10, y1 - 10), h - base - 5)
    cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3)
    cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


class EnvironmentAnalyzer:
    """Feed frames in time order with process_batch(); read .events at the end."""

    def __init__(self) -> None:
        self.model = get_environment_model()
        self.detector = EnvironmentalDetector(self.model.names)
        self.events: list[EnvironmentEvent] = []
        self._last_event_time: dict[str, float] = {}
        self.frames_analyzed = 0
        self.frames_failed = 0

    @property
    def model_name(self) -> str:
        return _model_name or get_settings().environment_model

    def process_batch(self, items: list[tuple[int, np.ndarray]]) -> None:
        """items: (offset_ms, BGR frame) in time order - one YOLO call for the whole batch."""
        if not items:
            return
        try:
            with _infer_lock:
                if (_model_name or "").endswith("_openvino_model"):
                    results = [self.model(frame, verbose=False, imgsz=list(OPENVINO_IMGSZ))[0] for _, frame in items]
                else:
                    results = self.model([frame for _, frame in items], verbose=False, imgsz=640)
        except Exception:
            log.exception("Environment detection failed on %s frames", len(items))
            self.frames_failed += len(items)
            return
        for (offset_ms, frame), result in zip(items, results):
            self.frames_analyzed += 1
            drawn, anomalies = self.detector.detect(frame.copy(), result)
            timestamp = offset_ms / 1000
            for anomaly in anomalies:
                previous = self._last_event_time.get(anomaly["type"])
                if previous is not None and timestamp - previous < EVENT_COOLDOWN_SECONDS:
                    continue
                self._last_event_time[anomaly["type"]] = timestamp
                self.events.append(EnvironmentEvent(timestamp, anomaly["type"], anomaly["confidence"],
                                                    encode_jpeg(drawn, quality=82, max_side=960)))

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for e in self.events:
            counts[e.type] = counts.get(e.type, 0) + 1
        return {"status": "completed", "model": self.model_name, "frames_analyzed": self.frames_analyzed,
                "frames_failed": self.frames_failed, "events_total": len(self.events), "events_by_type": counts,
                "events": [e.as_dict() for e in self.events]}
