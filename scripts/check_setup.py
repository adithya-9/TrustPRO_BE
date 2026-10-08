"""Verify the local setup: database, schema version, model files and the OpenCV build.

    python -m scripts.check_setup
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ok = True
    from app.core.config import get_settings

    settings = get_settings()

    import cv2

    has_yunet = hasattr(cv2, "FaceDetectorYN")
    print(f"[{'ok' if has_yunet else 'FAIL'}] OpenCV {cv2.__version__} (FaceDetectorYN available: {has_yunet})")
    ok &= has_yunet

    from sqlalchemy import create_engine, text

    try:
        with create_engine(settings.database_url).connect() as conn:
            version = conn.execute(text(f"SELECT version_num FROM {settings.db_schema}.trustpro_alembic_version")).scalar()
        print(f"[ok] Database reachable, TrustPRO schema version {version}")
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] Database / migrations: {exc.__class__.__name__}: {str(exc).splitlines()[0]}")
        print("       Run: alembic upgrade head")
        ok = False

    required = ["face_detection_yunet_2023mar.onnx", "arcface_w600k_r50.onnx", settings.environment_model_fallback]
    for name in required:
        exists = (settings.models_dir / name).exists()
        print(f"[{'ok' if exists else 'FAIL'}] model {name}")
        ok &= exists
    if (settings.models_dir / settings.environment_model).exists():
        print(f"[ok] model {settings.environment_model} (faster CPU version)")
    else:
        print(f"[warn] {settings.environment_model} missing - reports use the slower .pt model. "
              "Run: python -m scripts.export_environment_model")
    try:
        import torch
        import ultralytics

        print(f"[ok] Ultralytics {ultralytics.__version__} / PyTorch {torch.__version__}")
    except ImportError as exc:
        print(f"[FAIL] Ultralytics / PyTorch not installed: {exc}")
        ok = False
    if not ok:
        print("\nSome checks failed. Model files: python -m scripts.download_models")
    else:
        print("\nAll checks passed.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
