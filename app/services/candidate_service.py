"""Candidate profile: first and last name, date of birth, mobile number, city and profile photo.

Only the name, date of birth and profile photo are used for identity checks; they are compared
with the government ID captured on camera when the report is generated.
"""
from __future__ import annotations

import logging
import re
from datetime import date

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ai.face import face_quality, get_face_engine
from app.core.errors import InvalidUpload
from app.db.models import Candidate, FileCategory, IdVerification, ProfileStatus, User
from app.schemas.common import media_url
from app.schemas.profile import ProfileOut
from app.services.storage import decode_image, read_image, store_image

log = logging.getLogger(__name__)
MIN_PROFILE_FACE_PX = 80
NAME_RE = re.compile(r"[A-Za-z][A-Za-z .'\-]*")
MOBILE_RE = re.compile(r"^\+?[0-9]{10,15}$")


def get_candidate(db: Session, user: User) -> Candidate:
    candidate = db.scalar(select(Candidate).where(Candidate.user_id == user.user_id))
    if candidate is None:  # accounts created outside the register flow
        candidate = Candidate(user_id=user.user_id)
        db.add(candidate)
        db.commit()
    return candidate


def latest_capture(db: Session, candidate: Candidate) -> IdVerification | None:
    return db.scalar(
        select(IdVerification).where(IdVerification.candidate_id == candidate.candidate_id)
        .order_by(IdVerification.verification_id.desc()).limit(1)
    )


def profile_out(candidate: Candidate) -> ProfileOut:
    return ProfileOut(
        candidate_id=candidate.candidate_id,
        first_name=candidate.first_name,
        last_name=candidate.last_name,
        date_of_birth=candidate.date_of_birth.isoformat() if candidate.date_of_birth else None,
        mobile_number=candidate.mobile_number,
        city=candidate.city,
        profile_photo_url=media_url(candidate.profile_photo_file_id),
        complete=candidate.profile_status == ProfileStatus.COMPLETE,
    )


def _invalid(field: str, message: str) -> InvalidUpload:
    return InvalidUpload(message, code="VALIDATION_ERROR", details={"fields": {field: message}})


def _clean_name(value: str, field: str, label: str) -> str:
    name = re.sub(r"\s+", " ", value or "").strip()
    if not 1 <= len(name) <= 75 or not NAME_RE.fullmatch(name):
        raise _invalid(field, f"Enter your {label} using letters only.")
    return name


def _validate_dob(dob: date) -> date:
    today = date.today()
    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    if age < 14 or age > 100:
        raise _invalid("date_of_birth", "Enter a valid date of birth.")
    return dob


def _clean_mobile(value: str) -> str:
    mobile = re.sub(r"[\s\-()]", "", value or "")
    if not MOBILE_RE.match(mobile):
        raise _invalid("mobile_number", "Enter a valid mobile number (10 to 15 digits).")
    return mobile


def _clean_city(value: str | None) -> str | None:
    city = re.sub(r"\s+", " ", value or "").strip()
    if not city:
        return None
    if len(city) > 100 or not re.fullmatch(r"[A-Za-z][A-Za-z .'\-]*", city):
        raise _invalid("city", "Enter a valid city name.")
    return city


def _check_profile_photo(image: np.ndarray) -> None:
    faces = get_face_engine().detect(image)
    if not faces:
        raise InvalidUpload("We could not find a face in your profile photo. Use a clear, front-facing photo.",
                            code="PROFILE_PHOTO_NO_FACE")
    if len(faces) > 1 and faces[1].area > faces[0].area * 0.25:
        raise InvalidUpload("Your profile photo shows more than one person. Use a photo of only yourself.",
                            code="PROFILE_PHOTO_MULTIPLE_FACES")
    if face_quality(image, faces[0])["size_px"] < MIN_PROFILE_FACE_PX:
        raise InvalidUpload("Your face is too small in the profile photo. Use a closer, front-facing photo.",
                            code="PROFILE_PHOTO_FACE_TOO_SMALL")


def save_profile(db: Session, user: User, *, first_name: str, last_name: str, date_of_birth: date,
                 mobile_number: str, city: str | None, profile_photo: bytes | None) -> Candidate:
    candidate = get_candidate(db, user)
    first = _clean_name(first_name, "first_name", "first name")
    last = _clean_name(last_name, "last_name", "last name")
    dob = _validate_dob(date_of_birth)
    mobile = _clean_mobile(mobile_number)
    clean_city = _clean_city(city)
    if profile_photo is None and candidate.profile_photo_file_id is None:
        raise InvalidUpload("A profile photo is required.", code="PROFILE_PHOTO_REQUIRED")

    if profile_photo is not None:   # validate before storing anything
        image = decode_image(profile_photo, "profile photo")
        _check_profile_photo(image)
        candidate.profile_photo_file_id = store_image(
            db, user.user_id, FileCategory.PROFILE_PHOTO, f"candidates/{candidate.candidate_id}", image).file_id

    candidate.first_name, candidate.last_name = first, last
    candidate.date_of_birth = dob
    candidate.mobile_number = mobile
    candidate.city = clean_city
    candidate.profile_status = ProfileStatus.COMPLETE
    db.commit()
    log.info("Profile saved candidate_id=%s", candidate.candidate_id)
    return candidate


def profile_embedding(candidate: Candidate) -> np.ndarray | None:
    """Face embedding of the profile photo. Computed on demand - templates are not stored."""
    if candidate.profile_photo is None:
        return None
    engine = get_face_engine()
    image = read_image(candidate.profile_photo)
    faces = engine.detect(image)
    return engine.embed(image, faces[0]) if faces else None


def require_complete_profile(candidate: Candidate) -> None:
    if candidate.profile_status != ProfileStatus.COMPLETE:
        raise InvalidUpload("Please complete your profile first.", code="PROFILE_INCOMPLETE", status_code=409)
