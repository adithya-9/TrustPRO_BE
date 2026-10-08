"""Government ID verification before the assessment.

The candidate shows the ID to the camera. The name, ID number, date of birth and ID photo are
read at once and shown back; the candidate saves them (stored as JSON on the verification
record) or retakes the picture. The assessment can start only after a saved verification.
Report generation later compares the ID with the profile and the assessment video.
"""
from __future__ import annotations

import logging
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.ai.id_document import DocumentReading, extract_details
from app.core.config import get_settings
from app.core.errors import Conflict, NotFound
from app.db.models import FileCategory, IdVerification, User, VerificationStatus, utcnow
from app.pipeline.identity import id_photo_label
from app.schemas.common import iso, media_url
from app.schemas.verification import CaptureOut, IdDetails
from app.services import candidate_service
from app.services.storage import decode_image, store_image

log = logging.getLogger(__name__)


def extract_and_store(db: Session, user: User, image_bytes: bytes) -> IdVerification:
    candidate = candidate_service.get_candidate(db, user)
    candidate_service.require_complete_profile(candidate)
    image = decode_image(image_bytes, "ID capture")
    attempts = db.scalar(select(func.count(IdVerification.verification_id))
                         .where(IdVerification.candidate_id == candidate.candidate_id)) or 0
    folder = f"candidates/{candidate.candidate_id}/id_verification"
    capture = store_image(db, user.user_id, FileCategory.ID_CAPTURE, folder, image, max_side=1600)
    details = extract_details(image, candidate.first_name or "", candidate.last_name or "")
    portrait = details.pop("portrait_crop")
    record = IdVerification(candidate_id=candidate.candidate_id, attempt_no=attempts + 1,
                            status=VerificationStatus.CAPTURED, capture_file_id=capture.file_id,
                            portrait_detected=portrait is not None,
                            extracted_fields={"details": details, "confirmed": False})
    if portrait is not None and portrait.size:
        record.portrait_file_id = store_image(db, user.user_id, FileCategory.ID_PORTRAIT, folder, portrait,
                                              max_side=600).file_id
    db.add(record)
    db.commit()
    log.info("ID captured candidate_id=%s attempt=%s portrait=%s", candidate.candidate_id, record.attempt_no,
             portrait is not None)   # ID details are personal data: not logged
    return record


def confirm(db: Session, user: User, verification_id: int) -> IdVerification:
    candidate = candidate_service.get_candidate(db, user)
    record = db.get(IdVerification, verification_id)
    if record is None or record.candidate_id != candidate.candidate_id:
        raise NotFound("ID verification not found.")
    latest = candidate_service.latest_capture(db, candidate)
    if latest is None or latest.verification_id != record.verification_id:
        raise Conflict("A newer ID picture was taken. Please save that one.", code="ID_NOT_LATEST")
    record.extracted_fields = {**(record.extracted_fields or {}), "confirmed": True, "confirmed_at": iso(utcnow())}
    db.commit()
    log.info("ID details saved candidate_id=%s verification_id=%s", candidate.candidate_id, verification_id)
    return record


def is_confirmed(record: IdVerification | None) -> bool:
    return bool(record and (record.extracted_fields or {}).get("confirmed"))


def capture_out(record: IdVerification) -> CaptureOut:
    fields = record.extracted_fields or {}
    return CaptureOut(verification_id=record.verification_id, attempt_no=record.attempt_no,
                      capture_url=media_url(record.capture_file_id), portrait_url=media_url(record.portrait_file_id),
                      details=IdDetails(**{k: v for k, v in (fields.get("details") or {}).items()
                                           if k in IdDetails.model_fields}),
                      confirmed=bool(fields.get("confirmed")), created_at=iso(record.created_at))


def save_analysis(db: Session, record: IdVerification, owner_user_id: int, reading: DocumentReading) -> dict:
    """Write the report-time analysis to the capture record; returns the summary for the report."""
    settings = get_settings()
    threshold = settings.id_photo_match_threshold
    if record.portrait_file_id is None and reading.portrait_crop is not None and reading.portrait_crop.size:
        portrait = store_image(db, owner_user_id, FileCategory.ID_PORTRAIT,
                               f"candidates/{record.candidate_id}/id_verification", reading.portrait_crop, max_side=600)
        record.portrait_file_id = portrait.file_id
    name = reading.name
    sim = reading.face_similarity_profile
    record.portrait_detected = reading.portrait is not None
    record.text_readable = reading.text_readable
    record.name_match_score = None if name is None else Decimal(str(round(name.score, 2)))
    record.name_matched = None if name is None else name.matched
    record.dob_match = reading.dob_match
    record.face_similarity_profile = None if sim is None else Decimal(str(round(sim, 4)))
    record.quality = reading.quality.as_dict()
    record.extracted_fields = {**(record.extracted_fields or {}), "analysis": reading.extracted_fields()}
    record.status = VerificationStatus.ANALYSED
    record.analysed_at = utcnow()
    db.flush()
    return {
        "verification_id": record.verification_id,
        "attempt_no": record.attempt_no,
        "capture_url": media_url(record.capture_file_id),
        "portrait_url": media_url(record.portrait_file_id),
        "captured_at": iso(record.created_at),
        "checks": {
            "portrait_detected": record.portrait_detected,
            "text_readable": record.text_readable,
            "text_lines_read": len(reading.lines),
            "name": name.as_dict() if name else None,
            "dob_match": reading.dob_match,
            "dates_found": reading.dates_found,
            "face_vs_profile": {"similarity": None if sim is None else round(sim, 4), "threshold": threshold,
                                "result": id_photo_label(sim),
                                "consistent": {"CONSISTENT": True, "NOT_CONSISTENT": False}.get(id_photo_label(sim))},
        },
        "quality": record.quality,
    }

