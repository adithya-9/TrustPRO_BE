"""The analysis steps shared by the live path (while the assessment is recorded, see live.py)
and the full sweep after the assessment (runner.py): input loading, the analyzers, the
government-ID check and the parallel frame loop.
"""
from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable, Iterable

import numpy as np

from app.ai.face import get_face_engine
from app.ai.id_document import read_document
from app.ai.runtime import ModelUnavailableError
from app.core.config import get_settings
from app.db.models import Interview, MediaFile
from app.db.session import session_scope
from app.pipeline.environment import BATCH_SIZE, EnvironmentAnalyzer
from app.pipeline.identity import IdentityAnalyzer
from app.services import candidate_service
from app.services.storage import get_storage, read_image

log = logging.getLogger(__name__)
_SENTINEL = object()


class ReportFailed(RuntimeError):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.user_message = message


def load_inputs(interview_id: int) -> dict:
    """Everything the analysis needs about the assessment, the candidate and their saved ID."""
    with session_scope() as db:
        interview = db.get(Interview, interview_id)
        candidate = interview.candidate
        if interview.recording is None:
            raise ReportFailed("No recording was saved for this assessment, so it cannot be analysed.")
        verification = interview.verification
        capture = db.get(MediaFile, verification.capture_file_id) if verification.capture_file_id else None
        portrait = db.get(MediaFile, verification.portrait_file_id) if verification.portrait_file_id else None
        profile_photo = db.get(MediaFile, candidate.profile_photo_file_id) if candidate.profile_photo_file_id else None
        video_path = get_storage().path_for(interview.recording.storage_key)
        return {
            "interview_id": interview.interview_id,
            "candidate_id": candidate.candidate_id,
            "owner_user_id": candidate.user_id,
            "duration_ms": interview.duration_ms or 0,
            "started_at": interview.started_at,
            "video_path": video_path,
            "folder": video_path.parent,
            "verification_id": interview.verification_id,
            "id_details": dict((verification.extracted_fields or {}).get("details") or {}),
            "candidate": {"first_name": candidate.first_name, "last_name": candidate.last_name,
                          "date_of_birth": candidate.date_of_birth, "mobile_number": candidate.mobile_number,
                          "city": candidate.city},
            "first_name": candidate.first_name or "",
            "last_name": candidate.last_name or "",
            "dob": candidate.date_of_birth,
            "profile_embedding": candidate_service.profile_embedding(candidate),
            "capture_image": read_image(capture) if capture else None,
            "portrait_image": read_image(portrait) if portrait else None,
            "profile_image": read_image(profile_photo) if profile_photo else None,
        }


def face_every(interval_ms: int) -> int:
    """Faces are compared on one sampled frame in this many (analysis_face_interval_s)."""
    return max(1, round(get_settings().analysis_face_interval_s * 1000 / interval_ms))


def make_analyzers(inputs: dict, interval_ms: int) -> tuple[IdentityAnalyzer | None, EnvironmentAnalyzer | None, dict]:
    """(identity, environment, failed sections). A missing model fails only its own section."""
    sections: dict[str, dict] = {}
    identity = environment = None
    try:
        identity = IdentityAnalyzer(get_face_engine(), inputs["profile_embedding"], interval_ms * face_every(interval_ms))
    except (ValueError, ModelUnavailableError) as exc:
        sections["identity"] = {"status": "failed", "error": str(exc) or "The face model is not available."}
    try:
        environment = EnvironmentAnalyzer()
    except ModelUnavailableError as exc:
        sections["environment"] = {"status": "failed", "error": "The object detection model is not available."}
        log.error("Environment model unavailable: %s", exc)
    return identity, environment, sections


def analyse_id_document(inputs: dict, result: dict) -> None:
    """OCR name/DOB and ID photo vs profile. Fills result["reading"] or result["summary"] (failure)."""
    if inputs["capture_image"] is None:
        result["summary"] = {"status": "failed", "error": "No government ID was captured before the assessment."}
        return
    try:
        result["reading"] = read_document(inputs["capture_image"], first_name=inputs["first_name"],
                                          last_name=inputs["last_name"], profile_dob=inputs["dob"],
                                          profile_embedding=inputs["profile_embedding"])
    except Exception:
        log.exception("ID document analysis failed interview_id=%s", inputs["interview_id"])
        result["summary"] = {"status": "failed", "error": "The government ID image could not be analysed."}


def analyse_frames(frames: Iterable[tuple[int, np.ndarray]], identity: IdentityAnalyzer | None,
                   environment: EnvironmentAnalyzer | None, every: int,
                   on_progress: Callable[[int], None] | None = None) -> BaseException | None:
    """Run identity (one frame in `every`) and environment (every frame, 8 per YOLO call) on the
    sampled frames in parallel. Returns the error that stopped frame reading, if any."""
    id_q: queue.Queue = queue.Queue(maxsize=24)
    env_q: queue.Queue = queue.Queue(maxsize=24)
    done = {"identity": 0, "environment": 0}
    done_lock = threading.Lock()
    producer_error: list[BaseException] = []

    def producer() -> None:
        try:
            for i, (offset_ms, frame) in enumerate(frames):
                if identity is not None and i % every == 0:
                    id_q.put((offset_ms, frame))
                if environment is not None:
                    env_q.put((offset_ms, frame))
        except BaseException as exc:  # noqa: BLE001 - returned to the caller
            producer_error.append(exc)
        finally:
            id_q.put(_SENTINEL)
            env_q.put(_SENTINEL)

    def report_progress() -> None:
        if on_progress is None:
            return
        with done_lock:
            n = min(done["identity"] if identity else done["environment"],
                    done["environment"] if environment else done["identity"])
        on_progress(n)

    def identity_worker() -> None:
        while (item := id_q.get()) is not _SENTINEL:
            identity.process(*item)
            with done_lock:
                done["identity"] += every
            report_progress()

    def environment_worker() -> None:
        batch: list = []
        while True:
            item = env_q.get()
            if item is not _SENTINEL:
                batch.append(item)
            if batch and (item is _SENTINEL or len(batch) >= BATCH_SIZE):
                environment.process_batch(batch)   # rules applied in time order
                with done_lock:
                    done["environment"] += len(batch)
                batch = []
                report_progress()
            if item is _SENTINEL:
                return

    threads = [threading.Thread(target=producer, name="frames", daemon=True)]
    if identity is not None:
        threads.append(threading.Thread(target=identity_worker, name="identity", daemon=True))
    if environment is not None:
        threads.append(threading.Thread(target=environment_worker, name="environment", daemon=True))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return producer_error[0] if producer_error else None
