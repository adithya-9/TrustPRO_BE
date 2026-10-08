from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict


class ApiModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


def iso(value: datetime | None) -> str | None:
    """Database timestamps are naive UTC; send them with an explicit offset."""
    if value is None:
        return None
    return value.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def media_url(file_id: int | None) -> str | None:
    return f"/api/media/{file_id}" if file_id else None


class Issue(ApiModel):
    code: str
    message: str


class MessageOut(ApiModel):
    message: str
