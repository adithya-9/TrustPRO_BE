"""Text recognition for government IDs with RapidOCR (PaddleOCR PP-OCRv5 models on ONNX Runtime).

PP-OCRv5 mobile detection + the English PP-OCRv5 recogniser read Latin-script ID cards
(Aadhaar, PAN, driving licence, passport data page) well on CPU in ~2-4 s per image.
The default Chinese+English recogniser was clearly worse on Indian ID cards in our tests.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TextLine:
    text: str
    confidence: float
    box: tuple[int, int, int, int]  # x, y, w, h


class OcrEngine:
    def __init__(self) -> None:
        from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR

        self._engine = RapidOCR(params={
            "Global.log_level": "warning",
            "Det.ocr_version": OCRVersion.PPOCRV5,
            "Det.model_type": ModelType.MOBILE,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Rec.model_type": ModelType.MOBILE,
            "Rec.lang_type": LangRec.EN,
        })
        self._lock = threading.Lock()

    def read(self, image: np.ndarray, target_side: int = 1600) -> list[TextLine]:
        h, w = image.shape[:2]
        scale = target_side / max(h, w)
        if abs(scale - 1.0) > 0.05:
            interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=interp)
        with self._lock:
            result = self._engine(image)
        lines: list[TextLine] = []
        for text, score, box in zip(result.txts or (), result.scores or (), result.boxes if result.boxes is not None else ()):
            pts = np.asarray(box, dtype=np.float32) / scale
            x, y = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            lines.append(TextLine(str(text).strip(), float(score), (int(x), int(y), int(x2 - x), int(y2 - y))))
        return [line for line in lines if line.text]


_instance: OcrEngine | None = None
_lock = threading.Lock()


def get_ocr_engine() -> OcrEngine:
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = OcrEngine()
    return _instance
