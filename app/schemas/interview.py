from __future__ import annotations

from pydantic import Field

from app.schemas.common import ApiModel


class ReportBrief(ApiModel):
    report_id: int
    status: str
    progress_pct: int
    current_stage: str | None
    error_message: str | None


class InterviewOut(ApiModel):
    interview_id: int
    status: str                    # CREATED | LIVE | ENDED | ABANDONED
    identity_status: str           # ID_CAPTURED (checked during report generation)
    started_at: str | None
    ended_at: str | None
    duration_ms: int | None
    recording_status: str          # NONE | UPLOADING | COMPLETE | FAILED
    recording_chunks: int
    recording_url: str | None
    created_at: str | None
    latest_report: ReportBrief | None


class EndInterviewRequest(ApiModel):
    duration_ms: int = Field(ge=0, le=6 * 60 * 60 * 1000)
    chunk_count: int = Field(ge=0, le=100_000)


class ChunkAck(ApiModel):
    next_seq: int
    bytes_received: int
