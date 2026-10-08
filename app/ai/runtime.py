"""Shared helpers for creating ONNX Runtime sessions.

Every model is CPU-first. GPU providers can be listed in ONNX_PROVIDERS; anything the
installed onnxruntime build does not support is skipped and CPU is always the fallback.
"""
from __future__ import annotations

import logging
from pathlib import Path

import onnxruntime as ort

from app.core.config import get_settings

log = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    """A model file is missing or could not be loaded."""


def model_path(filename: str) -> Path:
    path = get_settings().models_dir / filename
    if not path.exists():
        raise ModelUnavailableError(
            f"Model file '{filename}' is missing. Run: python -m scripts.download_models"
        )
    return path


def create_session(filename: str, threads: int | None = None) -> ort.InferenceSession:
    settings = get_settings()
    available = set(ort.get_available_providers())
    providers = [p for p in settings.onnx_providers if p in available]
    if "CPUExecutionProvider" not in providers:
        providers.append("CPUExecutionProvider")

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads or settings.inference_threads
    options.inter_op_num_threads = 1
    # Several sessions run concurrently (live monitors + report jobs). Spin-waiting threads
    # would burn cores the other sessions need, so idle threads sleep instead.
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    try:
        session = ort.InferenceSession(str(model_path(filename)), options, providers=providers)
    except ModelUnavailableError:
        raise
    except Exception as exc:  # corrupted file, unsupported opset, ...
        raise ModelUnavailableError(f"Model '{filename}' could not be loaded: {exc}") from exc
    log.info("Loaded model %s with providers %s", filename, session.get_providers())
    return session
