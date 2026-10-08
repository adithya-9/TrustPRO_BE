"""Test fixtures.

Integration tests use the real trustmate database (DATABASE_URL) - no separate database is
created. Every test user gets a unique email and is deleted afterwards; deleting the user
cascades to all TrustPRO rows. Files go to a temporary STORAGE_DIR.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import pytest

_storage = Path(tempfile.mkdtemp(prefix="trustpro-test-storage-"))
os.environ["STORAGE_DIR"] = str(_storage)

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import delete, text  # noqa: E402

from app.db.models import (Candidate, FileCategory, IdVerification, ProfileStatus, User,  # noqa: E402
                           VerificationStatus)
from app.db.session import SessionLocal  # noqa: E402
from app.main import app  # noqa: E402
from app.services.storage import store_image  # noqa: E402

PASSWORD = "Testpass123"


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c
    shutil.rmtree(_storage, ignore_errors=True)


@pytest.fixture()
def db():
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture()
def make_user(client):
    created: list[str] = []

    def _make(signed_in: bool = True) -> tuple[TestClient, str]:
        email = f"test-{uuid.uuid4().hex[:12]}@example.com"
        created.append(email)
        c = TestClient(app)
        # Share the session client's event loop. A per-request loop would be torn down as
        # soon as a WebSocket closes, cutting off the live monitor's clean-up (which the real
        # server lets finish).
        c.portal = client.portal
        r = c.post("/api/auth/register", json={"email": email, "password": PASSWORD})
        assert r.status_code == 201, r.text
        if not signed_in:
            c.cookies.clear()
        return c, email

    yield _make
    with SessionLocal() as session:
        # Fail fast instead of hanging if a background job still holds a row lock.
        session.execute(text("SET LOCAL lock_timeout = '15s'"))
        session.execute(delete(User).where(User.email.in_(created)))
        session.commit()


def synthetic_image(width: int = 640, height: int = 480, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(60, 200, size=(height, width, 3), dtype=np.uint8)


def jpeg_bytes(image: np.ndarray) -> bytes:
    return cv2.imencode(".jpg", image)[1].tobytes()


@pytest.fixture()
def verified_candidate(make_user):
    """A signed-in candidate whose profile and ID capture are inserted directly.

    Real face/ID images are deliberately not stored in the repository, so the AI-dependent
    onboarding steps are bypassed here; they are covered by unit tests and by
    manual testing with real media."""
    c, email = make_user()
    with SessionLocal() as session:
        user = session.query(User).filter_by(email=email).one()
        cand = session.query(Candidate).filter_by(user_id=user.user_id).one()
        photo = store_image(session, user.user_id, FileCategory.PROFILE_PHOTO, "test", synthetic_image(seed=1))
        capture = store_image(session, user.user_id, FileCategory.ID_CAPTURE, "test", synthetic_image(seed=2))
        cand.first_name, cand.last_name, cand.date_of_birth = "Test", "Candidate", date(1995, 1, 1)
        cand.mobile_number, cand.profile_photo_file_id = "9876543210", photo.file_id
        cand.profile_status = ProfileStatus.COMPLETE
        session.flush()
        session.add(IdVerification(candidate_id=cand.candidate_id, attempt_no=1, status=VerificationStatus.CAPTURED,
                                   capture_file_id=capture.file_id,
                                   extracted_fields={"details": {"name": "Test Candidate", "id_number": None,
                                                                 "id_type": None, "dob": "1995-01-01"},
                                                     "confirmed": True}))
        session.commit()
    return c


def synthetic_webm(path: Path, seconds: float = 6.0, fps: int = 10) -> int:
    """A small WebM with no people in it. Returns the duration in ms."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"VP80"), fps, (320, 240))
    frames = int(seconds * fps)
    ramp = np.tile(np.linspace(30, 200, 320, dtype=np.float32), (240, 1))
    for i in range(frames):
        shade = np.roll(ramp, i * 4, axis=1).astype(np.uint8)   # a slowly moving gradient
        writer.write(cv2.merge([shade, shade, shade]))
    writer.release()
    return int(frames / fps * 1000)
