"""Interview lifecycle and recording storage.

The browser records with MediaRecorder and uploads the WebM stream in small sequential
chunks while the interview is running, so a network hiccup or closed tab loses seconds,
not the whole interview, and nothing large is held in browser memory.
"""
from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.errors import Conflict, Forbidden, InvalidUpload, NotFound
from app.db.models import (FileCategory, Interview, InterviewReport, InterviewStatus, MediaFile, RecordingStatus,
                           ReportStatus, User, utcnow)
from app.schemas.common import iso
from app.schemas.interview import InterviewOut, ReportBrief
from app.services import candidate_service
from app.services.storage import get_storage

log = logging.getLogger(__name__)

STATUS = {InterviewStatus.CREATED: "CREATED", InterviewStatus.LIVE: "LIVE",
          InterviewStatus.ENDED: "ENDED", InterviewStatus.ABANDONED: "ABANDONED"}
RECORDING = {RecordingStatus.NONE: "NONE", RecordingStatus.UPLOADING: "UPLOADING",
             RecordingStatus.COMPLETE: "COMPLETE", RecordingStatus.FAILED: "FAILED"}
REPORT = {ReportStatus.QUEUED: "QUEUED", ReportStatus.PROCESSING: "PROCESSING",
          ReportStatus.COMPLETED: "COMPLETED", ReportStatus.FAILED: "FAILED"}
WEBM_MAGIC = b"\x1a\x45\xdf\xa3"


def _follow_recording(interview: Interview) -> None:
    """Analyse the recording while it is uploaded (app/pipeline/live.py); idempotent. Started on
    every chunk too, so a server restart mid-assessment picks the recording up again."""
    from app.pipeline.live import registry  # local import: keeps the API importable without loading models

    registry.ensure(interview.interview_id, int(get_settings().analysis_sample_interval_s * 1000))


def get_owned_interview(db: Session, user: User, interview_id: int) -> Interview:
    interview = db.get(Interview, interview_id)
    if interview is None:
        raise NotFound("Interview not found.")
    if interview.candidate.user_id != user.user_id:
        raise Forbidden("You do not have access to this interview.")
    return interview


def latest_report(db: Session, interview_id: int) -> InterviewReport | None:
    return db.scalar(select(InterviewReport).where(InterviewReport.interview_id == interview_id)
                     .order_by(InterviewReport.report_id.desc()).limit(1))


def interview_out(db: Session, interview: Interview) -> InterviewOut:
    report = latest_report(db, interview.interview_id)
    return InterviewOut(
        interview_id=interview.interview_id,
        status=STATUS[interview.status],
        identity_status="ID_CAPTURED",   # checked against the profile when the report is generated
        started_at=iso(interview.started_at),
        ended_at=iso(interview.ended_at),
        duration_ms=interview.duration_ms,
        recording_status=RECORDING[interview.recording_status],
        recording_chunks=interview.recording_chunks,
        recording_url=f"/api/interviews/{interview.interview_id}/recording"
        if interview.recording_status == RecordingStatus.COMPLETE else None,
        created_at=iso(interview.created_at),
        latest_report=ReportBrief(
            report_id=report.report_id, status=REPORT[report.status], progress_pct=report.progress_pct,
            current_stage=report.current_stage, error_message=report.error_message,
        ) if report else None,
    )


def list_interviews(db: Session, user: User) -> list[Interview]:
    candidate = candidate_service.get_candidate(db, user)
    return list(db.scalars(select(Interview).where(Interview.candidate_id == candidate.candidate_id)
                           .order_by(Interview.interview_id.desc())))


def create_interview(db: Session, user: User) -> Interview:
    candidate = candidate_service.get_candidate(db, user)
    candidate_service.require_complete_profile(candidate)
    capture = candidate_service.latest_capture(db, candidate)
    from app.services.verification_service import is_confirmed

    if not is_confirmed(capture):
        raise Conflict("Please show your government ID and save the details before starting the assessment.",
                       code="ID_NOT_CONFIRMED")
    interview = Interview(candidate_id=candidate.candidate_id, verification_id=capture.verification_id)
    db.add(interview)
    db.commit()
    log.info("Interview created interview_id=%s candidate_id=%s", interview.interview_id, candidate.candidate_id)
    return interview


def start_interview(db: Session, interview: Interview) -> Interview:
    if interview.status == InterviewStatus.LIVE:
        return interview
    if interview.status != InterviewStatus.CREATED:
        raise Conflict("This interview has already finished.", code="INTERVIEW_FINISHED")
    storage = get_storage()
    # Recordings are kept per candidate: storage/candidates/<candidate id>/assessments/<assessment id>/
    key = storage.new_key(f"candidates/{interview.candidate_id}/assessments/{interview.interview_id}", ".webm")
    media = MediaFile(owner_user_id=interview.candidate.user_id, file_category=FileCategory.RECORDING,
                      storage_key=key, content_type="video/webm", size_bytes=0)
    db.add(media)
    db.flush()
    interview.recording_file_id = media.file_id
    interview.recording_status = RecordingStatus.UPLOADING
    interview.recording_chunks = 0
    interview.status = InterviewStatus.LIVE
    interview.started_at = utcnow()
    db.commit()
    log.info("Interview started interview_id=%s", interview.interview_id)
    _follow_recording(interview)
    return interview


def append_chunk(db: Session, interview: Interview, seq: int, data: bytes) -> tuple[int, int]:
    settings = get_settings()
    if interview.status not in (InterviewStatus.LIVE, InterviewStatus.ENDED) or interview.recording is None:
        raise Conflict("The interview is not recording.", code="NOT_RECORDING")
    if interview.recording_status == RecordingStatus.COMPLETE:
        raise Conflict("The recording is already complete.", code="RECORDING_COMPLETE")
    if seq < interview.recording_chunks:
        # A retry of a chunk we already stored (the response was lost). Acknowledge it.
        return interview.recording_chunks, interview.recording.size_bytes
    if seq > interview.recording_chunks:
        raise Conflict("A part of the recording is missing.", code="CHUNK_OUT_OF_ORDER",
                       details={"expected_seq": interview.recording_chunks})
    if not data:
        raise InvalidUpload("Empty recording chunk.", code="EMPTY_CHUNK")
    if len(data) > settings.max_video_chunk_bytes:
        raise InvalidUpload("Recording chunk is too large.", code="CHUNK_TOO_LARGE")
    if seq == 0 and not data.startswith(WEBM_MAGIC):
        raise InvalidUpload("The recording must be a WebM video.", code="INVALID_VIDEO")
    if interview.recording.size_bytes + len(data) > settings.max_video_bytes:
        raise InvalidUpload("The recording is larger than the allowed maximum.", code="RECORDING_TOO_LARGE")

    size = get_storage().append(interview.recording.storage_key, data)
    interview.recording.size_bytes = size
    interview.recording_chunks = seq + 1
    db.commit()
    if interview.status == InterviewStatus.LIVE:
        _follow_recording(interview)
    return interview.recording_chunks, size


def end_interview(db: Session, interview: Interview, duration_ms: int, chunk_count: int) -> Interview:
    if interview.status == InterviewStatus.CREATED:
        raise Conflict("The interview has not started.", code="INTERVIEW_NOT_STARTED")
    if chunk_count != interview.recording_chunks:
        raise Conflict("Some of the recording is still uploading.", code="RECORDING_INCOMPLETE",
                       details={"received_chunks": interview.recording_chunks, "expected_chunks": chunk_count})
    if interview.status == InterviewStatus.LIVE:
        interview.status = InterviewStatus.ENDED
        interview.ended_at = utcnow()
        wall_ms = int((interview.ended_at - interview.started_at).total_seconds() * 1000)
        # The browser's duration is the recording length; never trust it beyond wall-clock time.
        interview.duration_ms = min(duration_ms, wall_ms + 5000)
    if interview.recording_status == RecordingStatus.UPLOADING:
        interview.recording_status = RecordingStatus.COMPLETE if chunk_count > 0 else RecordingStatus.FAILED
    db.commit()
    log.info("Interview ended interview_id=%s duration_ms=%s chunks=%s bytes=%s", interview.interview_id,
             interview.duration_ms, interview.recording_chunks, interview.recording.size_bytes if interview.recording else 0)
    from app.pipeline.live import registry

    if interview.recording_status == RecordingStatus.COMPLETE:
        registry.finish(interview.interview_id)     # the file is complete: analyse the last seconds now
        queue_report(db, interview)
    else:
        registry.cancel(interview.interview_id)
    return interview


def queue_report(db: Session, interview: Interview) -> InterviewReport:
    """The report is generated on the server right after the assessment; it is not shown to the
    candidate. When it is ready the server log prints a link to the HTML report."""
    existing = latest_report(db, interview.interview_id)
    if existing and existing.status in (ReportStatus.QUEUED, ReportStatus.PROCESSING, ReportStatus.COMPLETED):
        return existing
    report = InterviewReport(interview_id=interview.interview_id, status=ReportStatus.QUEUED, current_stage="Queued",
                             sample_interval_ms=int(get_settings().analysis_sample_interval_s * 1000))
    db.add(report)
    db.commit()
    from app.pipeline.jobs import jobs  # local import: keeps the API importable without loading models

    jobs.submit(report.report_id)
    return report
