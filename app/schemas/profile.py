from __future__ import annotations

from app.schemas.common import ApiModel


class ProfileOut(ApiModel):
    candidate_id: int
    first_name: str | None
    last_name: str | None
    date_of_birth: str | None
    mobile_number: str | None
    city: str | None
    profile_photo_url: str | None
    complete: bool
