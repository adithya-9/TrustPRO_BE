from __future__ import annotations

from app.schemas.common import ApiModel


class IdDetails(ApiModel):
    """What was read from the government ID; shown to the candidate to save or retake."""
    name: str | None = None
    id_number: str | None = None
    id_type: str | None = None
    dob: str | None = None


class CaptureOut(ApiModel):
    verification_id: int
    attempt_no: int
    capture_url: str | None
    portrait_url: str | None = None
    details: IdDetails = IdDetails()
    confirmed: bool = False
    created_at: str | None
