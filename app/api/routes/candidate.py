from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.api.deps import current_user
from app.core.config import get_settings
from app.core.errors import InvalidUpload
from app.db.models import User
from app.db.session import get_db
from app.schemas.profile import ProfileOut
from app.schemas.verification import CaptureOut
from app.services import candidate_service, verification_service

router = APIRouter(prefix="/api/candidate", tags=["Candidate"])


async def _read_upload(upload: UploadFile | None, label: str) -> bytes | None:
    if upload is None or not upload.filename:
        return None
    limit = get_settings().max_image_bytes
    data = await upload.read(limit + 1)
    if len(data) > limit:
        raise InvalidUpload(f"The {label} is too large. The maximum size is {limit // (1024 * 1024)} MB.")
    return data


@router.get("/profile", response_model=ProfileOut, summary="The signed-in candidate's profile")
def get_profile(user: User = Depends(current_user), db: Session = Depends(get_db)) -> ProfileOut:
    return candidate_service.profile_out(candidate_service.get_candidate(db, user))


@router.put("/profile", response_model=ProfileOut, summary="Create or update the candidate profile",
            description="Multipart form. The profile photo is required on first save and optional afterwards.")
async def save_profile(
    first_name: str = Form(..., max_length=75),
    last_name: str = Form(..., max_length=75),
    date_of_birth: date = Form(...),
    mobile_number: str = Form(..., max_length=25),
    city: str | None = Form(None, max_length=100),
    profile_photo: UploadFile | None = File(None),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> ProfileOut:
    photo = await _read_upload(profile_photo, "profile photo")
    # Face detection is CPU-bound: keep it off the event loop.
    candidate = await run_in_threadpool(
        candidate_service.save_profile, db, user, first_name=first_name, last_name=last_name,
        date_of_birth=date_of_birth, mobile_number=mobile_number, city=city, profile_photo=photo,
    )
    return candidate_service.profile_out(candidate)


@router.post("/id-verifications", response_model=CaptureOut, status_code=201,
             summary="Read a camera picture of the government ID: name, ID number, DOB and ID photo")
async def capture_id(capture: UploadFile = File(...), user: User = Depends(current_user),
                     db: Session = Depends(get_db)) -> CaptureOut:
    data = await _read_upload(capture, "ID capture")
    if data is None:
        raise InvalidUpload("No ID image was received. Please capture your ID again.")
    record = await run_in_threadpool(verification_service.extract_and_store, db, user, data)
    return verification_service.capture_out(record)


@router.post("/id-verifications/{verification_id}/confirm", response_model=CaptureOut,
             summary="Save the ID details shown to the candidate")
def confirm_id(verification_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)) -> CaptureOut:
    return verification_service.capture_out(verification_service.confirm(db, user, verification_id))


@router.get("/id-verifications/latest", response_model=CaptureOut | None, summary="Most recent ID capture")
def latest_capture(user: User = Depends(current_user), db: Session = Depends(get_db)) -> CaptureOut | None:
    record = candidate_service.latest_capture(db, candidate_service.get_candidate(db, user))
    return verification_service.capture_out(record) if record else None
