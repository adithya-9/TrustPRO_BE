"""Face detection (YuNet) and face embeddings (ArcFace or SFace).

Detection always uses YuNet (OpenCV Zoo, MIT). It returns a box plus five landmarks
(eyes, nose tip, mouth corners), which is exactly what both recognisers need for alignment.

Recognition is selectable with FACE_ENGINE:
  * "arcface" - ArcFace ResNet-50 trained on WebFace600K (InsightFace buffalo_l pack).
                Clearly better separation in our tests, but the weights are licensed for
                non-commercial research only. Production use needs an InsightFace licence.
  * "sface"   - SFace (OpenCV Zoo, Apache-2.0). Commercially clean, weaker separation.
Both produce L2-normalised embeddings compared with cosine similarity.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

import cv2
import numpy as np

from app.ai.runtime import create_session, model_path
from app.core.config import get_settings

YUNET_FILE = "face_detection_yunet_2023mar.onnx"
SFACE_FILE = "face_recognition_sface_2021dec.onnx"
ARCFACE_FILE = "arcface_w600k_r50.onnx"

# Standard 112x112 ArcFace alignment template (InsightFace), five points in the same
# order YuNet reports: right eye, left eye, nose tip, right mouth corner, left mouth corner.
ARCFACE_TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)


@dataclass
class Face:
    box: tuple[int, int, int, int]       # x, y, w, h
    landmarks: np.ndarray                 # (5, 2)
    score: float
    raw: np.ndarray                       # YuNet row (needed by SFace alignCrop)

    @property
    def area(self) -> int:
        return self.box[2] * self.box[3]

    def box_dict(self) -> dict:
        x, y, w, h = self.box
        return {"x": x, "y": y, "w": w, "h": h}


class FaceEngine:
    """Thread-safe wrapper. YuNet/SFace objects in OpenCV are not re-entrant, so they are
    guarded by a lock; the ONNX Runtime ArcFace session is safe to share."""

    def __init__(self) -> None:
        settings = get_settings()
        self.engine_name = settings.face_engine
        self.match_threshold = (
            settings.arcface_match_threshold if self.engine_name == "arcface" else settings.sface_match_threshold
        )
        self._detector = cv2.FaceDetectorYN.create(
            str(model_path(YUNET_FILE)), "", (320, 320), settings.face_detect_score, 0.3, 5000
        )
        self._det_lock = threading.Lock()
        self._rec_lock = threading.Lock()
        if self.engine_name == "arcface":
            self._arcface = create_session(ARCFACE_FILE)
            self._arcface_input = self._arcface.get_inputs()[0].name
            self._sface = None
        elif self.engine_name == "sface":
            self._sface = cv2.FaceRecognizerSF.create(str(model_path(SFACE_FILE)), "")
            self._arcface = None
        else:
            raise ValueError(f"Unknown FACE_ENGINE '{self.engine_name}' (expected 'arcface' or 'sface')")

    # ---------------- detection ----------------
    def detect(self, image: np.ndarray, max_side: int = 960, min_score: float | None = None) -> list[Face]:
        h, w = image.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        work = cv2.resize(image, (int(w * scale), int(h * scale))) if scale < 1.0 else image
        with self._det_lock:
            self._detector.setInputSize((work.shape[1], work.shape[0]))
            if min_score is not None:
                self._detector.setScoreThreshold(min_score)
            try:
                _, rows = self._detector.detect(work)
            finally:
                if min_score is not None:
                    self._detector.setScoreThreshold(get_settings().face_detect_score)
        faces: list[Face] = []
        for row in rows if rows is not None else []:
            row = row.copy()
            row[:14] /= scale
            x, y, bw, bh = row[:4]
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(w, int(x + bw)), min(h, int(y + bh))
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            faces.append(Face((x1, y1, x2 - x1, y2 - y1), row[4:14].reshape(5, 2), float(row[14]), row))
        faces.sort(key=lambda f: f.area, reverse=True)
        return faces

    # ---------------- recognition ----------------
    def embed(self, image: np.ndarray, face: Face) -> np.ndarray:
        if self._arcface is not None:
            matrix, _ = cv2.estimateAffinePartial2D(face.landmarks.astype(np.float32), ARCFACE_TEMPLATE, method=cv2.LMEDS)
            aligned = cv2.warpAffine(image, matrix, (112, 112), borderValue=0.0)
            blob = (cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32) - 127.5) / 127.5
            vector = self._arcface.run(None, {self._arcface_input: blob.transpose(2, 0, 1)[None]})[0][0]
        else:
            with self._rec_lock:
                aligned = self._sface.alignCrop(image, face.raw)
                vector = self._sface.feature(aligned).flatten()
        vector = vector.astype(np.float32)
        return vector / (np.linalg.norm(vector) + 1e-9)

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.clip(np.dot(a, b), -1.0, 1.0))

    def is_match(self, similarity: float) -> bool:
        return similarity >= self.match_threshold


def crop_face(image: np.ndarray, face: Face, margin: float = 0.35) -> np.ndarray:
    h, w = image.shape[:2]
    x, y, bw, bh = face.box
    mx, my = int(bw * margin), int(bh * margin)
    return image[max(0, y - my):min(h, y + bh + my), max(0, x - mx):min(w, x + bw + mx)]


def face_quality(image: np.ndarray, face: Face) -> dict:
    """Simple, explainable quality signals for a face crop."""
    crop = cv2.cvtColor(crop_face(image, face, 0.0), cv2.COLOR_BGR2GRAY)
    if crop.size == 0:
        return {"sharpness": 0.0, "brightness": 0.0, "size_px": 0}
    return {
        "sharpness": round(float(cv2.Laplacian(crop, cv2.CV_64F).var()), 1),
        "brightness": round(float(crop.mean()), 1),
        "size_px": int(min(face.box[2], face.box[3])),
    }


_instance: FaceEngine | None = None
_lock = threading.Lock()


def get_face_engine() -> FaceEngine:
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = FaceEngine()
    return _instance
