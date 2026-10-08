"""Export the environment model (YOLO11m) to OpenVINO for faster CPU inference.

Same weights and same results (FP32); only the inference engine changes. Writes
models/yolo11m_openvino_model/, which the report pipeline uses when present (else yolo11m.pt).

    python -m scripts.export_environment_model
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from app.core.config import get_settings
from app.pipeline.environment import OPENVINO_IMGSZ


def main() -> int:
    from ultralytics import YOLO

    models = get_settings().models_dir
    source = models / "yolo11m.pt"
    if not source.exists():
        print(f"Missing {source}; run scripts/download_models.py first.")
        return 1
    # Fixed 384x640 input, one frame per call: the same letterboxed input PyTorch uses for the
    # 16:9 (1280x720) recordings. Measured on the dev laptop: ~225 ms vs ~1000 ms per frame with
    # PyTorch, same detections apart from +-0.02 confidence near the thresholds. A dynamic-shape
    # export and a 640x640 batch-8 export were both slower than PyTorch, so they are not used.
    out = Path(YOLO(str(source)).export(format="openvino", imgsz=list(OPENVINO_IMGSZ), batch=1, dynamic=False,
                                        half=False, int8=False))
    target = models / "yolo11m_openvino_model"
    if out.resolve() != target.resolve():
        shutil.rmtree(target, ignore_errors=True)
        shutil.move(str(out), str(target))
    print(f"Exported: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
