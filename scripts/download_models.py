"""Download the pretrained model files TrustPRO needs into MODELS_DIR.

    python -m scripts.download_models

Models (all CPU): YuNet face detection (MIT), ArcFace R50 face recognition (InsightFace weights:
non-commercial research only), YOLO11m for environmental anomalies (Ultralytics, AGPL-3.0) and
RapidOCR / PP-OCRv5 for the ID card (fetches its own weights on first use, warmed up here).
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402

# name -> (url, sha384 or None)
MODELS = {
    # YuNet face detector, OpenCV Zoo (MIT).
    "face_detection_yunet_2023mar.onnx": ("https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx", None),
}

ARCFACE_PACK = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
ARCFACE_MEMBER = "w600k_r50.onnx"
ARCFACE_FILE = "arcface_w600k_r50.onnx"
ARCFACE_SHA256 = "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43"


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    print(f"  downloading {target.name} ...", flush=True)
    with urllib.request.urlopen(url, timeout=120) as response, open(tmp, "wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
    tmp.replace(target)


def fetch_arcface(models_dir: Path) -> None:
    target = models_dir / ARCFACE_FILE
    if target.exists():
        print(f"  {ARCFACE_FILE} already present")
        return
    pack = models_dir / "buffalo_l.zip"
    download(ARCFACE_PACK, pack)
    with zipfile.ZipFile(pack) as zf:
        member = next(n for n in zf.namelist() if n.endswith(ARCFACE_MEMBER))
        target.write_bytes(zf.read(member))
    pack.unlink()
    if hashlib.sha256(target.read_bytes()).hexdigest() != ARCFACE_SHA256:
        target.unlink()
        raise SystemExit(f"Checksum mismatch for {ARCFACE_FILE}; download aborted.")


def main() -> None:
    settings = get_settings()
    models_dir = settings.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model folder: {models_dir}")
    for name, (url, sha384) in MODELS.items():
        target = models_dir / name
        if not (target.exists() and target.stat().st_size > 0):
            download(url, target)
        else:
            print(f"  {name} already present")
        if sha384 and hashlib.sha384(target.read_bytes()).hexdigest() != sha384:
            target.unlink()
            raise SystemExit(f"Checksum mismatch for {name}; the file was removed. Run the script again.")
    fetch_arcface(models_dir)

    yolo = models_dir / settings.environment_model_fallback
    if not yolo.exists():
        print(f"  downloading {yolo.name} (Ultralytics) ...", flush=True)
        from ultralytics.utils.downloads import attempt_download_asset

        downloaded = Path(attempt_download_asset(yolo.name))
        downloaded.replace(yolo)
    else:
        print(f"  {yolo.name} already present")
    if not (models_dir / settings.environment_model).exists():
        print(f"  exporting {settings.environment_model} (OpenVINO, faster on CPU) ...", flush=True)
        from scripts.export_environment_model import main as export_environment_model

        export_environment_model()
    else:
        print(f"  {settings.environment_model} already present")

    print("  warming up RapidOCR (downloads its PP-OCR weights on first use) ...", flush=True)
    import numpy as np

    from app.ai.ocr import get_ocr_engine

    get_ocr_engine().read(np.full((64, 256, 3), 255, dtype=np.uint8))
    print("All models ready.")

if __name__ == "__main__":
    main()
