"""Logging setup. Logs carry ids (user, candidate, interview, report), never personal data."""
from __future__ import annotations

import logging
import os
import sys


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if getattr(root, "_trustpro_configured", False):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    root._trustpro_configured = True  # type: ignore[attr-defined]

    # Quieten chatty third-party loggers.
    for name in ("uvicorn.access", "RapidOCR", "multipart", "PIL"):
        logging.getLogger(name).setLevel(logging.WARNING)
    # MediaPipe / absl / TFLite C++ logs.
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import cv2

        cv2.setLogLevel(2)  # errors only; OpenCV 5 warns about DNN targets on every model load
    except Exception:  # pragma: no cover
        pass
