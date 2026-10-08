"""ORM mappings for the TrustPRO tables in trustmate.epsoft.

The schema itself is owned by the Alembic migrations (raw SQL); these classes mirror it.
Status columns use the one-letter codes documented in the migration and in the enums below.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

from sqlalchemy import (BigInteger, Boolean, Computed, Date, DateTime, ForeignKey, Integer,
                        MetaData, Numeric, SmallInteger, String)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.core.config import get_settings


def utcnow() -> datetime:
    """Naive UTC timestamp (the columns are TIMESTAMP holding UTC, per house convention)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    metadata = MetaData(schema=get_settings().db_schema)


# ----------------------------------------------------------------- status codes
class UserType:
    CANDIDATE, REVIEWER, ADMIN = "C", "R", "A"


class FileCategory:
    PROFILE_PHOTO, ID_DOCUMENT, ID_CAPTURE, ID_PORTRAIT, RECORDING, EVIDENCE = "P", "I", "C", "T", "V", "E"


class ProfileStatus:
    INCOMPLETE, COMPLETE = "I", "C"


class VerificationStatus:
    CAPTURED, ANALYSED = "C", "A"


class InterviewStatus:
    CREATED, LIVE, ENDED, ABANDONED = "C", "L", "E", "A"


class RecordingStatus:
    NONE, UPLOADING, COMPLETE, FAILED = "N", "U", "C", "F"


class ReportStatus:
    QUEUED, PROCESSING, COMPLETED, FAILED = "Q", "P", "C", "F"


class EventSource:
    LIVE, ANALYSIS = "L", "A"


# ----------------------------------------------------------------- tables
class User(Base):
    __tablename__ = "users"

    user_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    user_type: Mapped[str] = mapped_column(String(1), default=UserType.CANDIDATE)
    failed_attempts: Mapped[int] = mapped_column(Integer, default=0)
    is_disabled: Mapped[bool] = mapped_column(Boolean, default=False)
    last_active: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    candidate: Mapped["Candidate | None"] = relationship(back_populates="user", uselist=False)


class UserSession(Base):
    __tablename__ = "user_sessions"

    session_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id", ondelete="CASCADE"))
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    user: Mapped[User] = relationship()


class MediaFile(Base):
    __tablename__ = "media_files"

    file_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    owner_user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id", ondelete="CASCADE"))
    file_category: Mapped[str] = mapped_column(String(1))
    storage_key: Mapped[str] = mapped_column(String(255), unique=True)
    content_type: Mapped[str] = mapped_column(String(100))
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    sha256: Mapped[str | None] = mapped_column(String(64))
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Candidate(Base):
    __tablename__ = "candidates"

    candidate_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id", ondelete="CASCADE"), unique=True)
    first_name: Mapped[str | None] = mapped_column(String(75))
    last_name: Mapped[str | None] = mapped_column(String(75))
    date_of_birth: Mapped[date | None] = mapped_column(Date)
    mobile_number: Mapped[str | None] = mapped_column(String(20))
    city: Mapped[str | None] = mapped_column(String(100))
    profile_photo_file_id: Mapped[int | None] = mapped_column(ForeignKey("media_files.file_id", ondelete="SET NULL"))
    profile_status: Mapped[str] = mapped_column(String(1), default=ProfileStatus.INCOMPLETE)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    user: Mapped[User] = relationship(back_populates="candidate")
    profile_photo: Mapped[MediaFile | None] = relationship(foreign_keys=[profile_photo_file_id])

    @property
    def full_name(self) -> str:
        return " ".join(part for part in (self.first_name, self.last_name) if part)


class IdVerification(Base):
    __tablename__ = "id_verifications"

    verification_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidates.candidate_id", ondelete="CASCADE"))
    attempt_no: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(1))
    capture_file_id: Mapped[int | None] = mapped_column(ForeignKey("media_files.file_id", ondelete="SET NULL"))
    portrait_file_id: Mapped[int | None] = mapped_column(ForeignKey("media_files.file_id", ondelete="SET NULL"))
    portrait_detected: Mapped[bool] = mapped_column(Boolean, default=False)
    text_readable: Mapped[bool] = mapped_column(Boolean, default=False)
    name_match_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    name_matched: Mapped[bool | None] = mapped_column(Boolean)
    dob_match: Mapped[bool | None] = mapped_column(Boolean)
    face_similarity_profile: Mapped[Decimal | None] = mapped_column(Numeric(6, 4))
    quality: Mapped[dict] = mapped_column(JSONB, default=dict)
    extracted_fields: Mapped[dict] = mapped_column(JSONB, default=dict)
    issues: Mapped[list] = mapped_column(JSONB, default=list)
    analysed_at: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Interview(Base):
    __tablename__ = "interviews"

    interview_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidates.candidate_id", ondelete="CASCADE"))
    verification_id: Mapped[int] = mapped_column(ForeignKey("id_verifications.verification_id"))
    status: Mapped[str] = mapped_column(String(1), default=InterviewStatus.CREATED)
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime)
    duration_ms: Mapped[int | None] = mapped_column(BigInteger)
    recording_file_id: Mapped[int | None] = mapped_column(ForeignKey("media_files.file_id", ondelete="SET NULL"))
    recording_status: Mapped[str] = mapped_column(String(1), default=RecordingStatus.NONE)
    recording_chunks: Mapped[int] = mapped_column(Integer, default=0)
    gaze_calibration: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    candidate: Mapped[Candidate] = relationship()
    verification: Mapped[IdVerification] = relationship()
    recording: Mapped[MediaFile | None] = relationship()


class GazeEvent(Base):
    __tablename__ = "interview_gaze_events"

    gaze_event_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    interview_id: Mapped[int] = mapped_column(ForeignKey("interviews.interview_id", ondelete="CASCADE"))
    direction: Mapped[str] = mapped_column(String(10))
    start_offset_ms: Mapped[int] = mapped_column(BigInteger)
    end_offset_ms: Mapped[int] = mapped_column(BigInteger)
    duration_ms: Mapped[int] = mapped_column(BigInteger, Computed("end_offset_ms - start_offset_ms"))
    avg_confidence: Mapped[Decimal | None] = mapped_column(Numeric(4, 3))
    frame_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class InterviewReport(Base):
    __tablename__ = "interview_reports"

    report_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    interview_id: Mapped[int] = mapped_column(ForeignKey("interviews.interview_id", ondelete="CASCADE"))
    status: Mapped[str] = mapped_column(String(1), default=ReportStatus.QUEUED)
    progress_pct: Mapped[int] = mapped_column(SmallInteger, default=0)
    current_stage: Mapped[str | None] = mapped_column(String(60))
    error_message: Mapped[str | None] = mapped_column(String(500))
    sample_interval_ms: Mapped[int] = mapped_column(Integer)
    frames_analyzed: Mapped[int] = mapped_column(Integer, default=0)
    identity_summary: Mapped[dict | None] = mapped_column(JSONB)
    environment_summary: Mapped[dict | None] = mapped_column(JSONB)
    gaze_summary: Mapped[dict | None] = mapped_column(JSONB)
    processing_info: Mapped[dict | None] = mapped_column(JSONB)
    requested_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    interview: Mapped[Interview] = relationship()


class ReportAccess(Base):
    """A recruiter login (users.user_type = 'R') that can open exactly one report until expires_at."""
    __tablename__ = "report_access"

    access_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    report_id: Mapped[int] = mapped_column(ForeignKey("interview_reports.report_id", ondelete="CASCADE"))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id", ondelete="CASCADE"), unique=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    report: Mapped[InterviewReport] = relationship()
    user: Mapped[User] = relationship()


class Evidence(Base):
    __tablename__ = "interview_evidence"

    evidence_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    interview_id: Mapped[int] = mapped_column(ForeignKey("interviews.interview_id", ondelete="CASCADE"))
    report_id: Mapped[int | None] = mapped_column(ForeignKey("interview_reports.report_id", ondelete="CASCADE"))
    evidence_kind: Mapped[str] = mapped_column(String(30))
    label: Mapped[str] = mapped_column(String(40))
    offset_ms: Mapped[int] = mapped_column(BigInteger)
    file_id: Mapped[int | None] = mapped_column(ForeignKey("media_files.file_id", ondelete="SET NULL"))
    confidence: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))
    model_result: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class EnvironmentEvent(Base):
    __tablename__ = "interview_environment_events"

    environment_event_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    interview_id: Mapped[int] = mapped_column(ForeignKey("interviews.interview_id", ondelete="CASCADE"))
    report_id: Mapped[int | None] = mapped_column(ForeignKey("interview_reports.report_id", ondelete="CASCADE"))
    source: Mapped[str] = mapped_column(String(1))
    event_type: Mapped[str] = mapped_column(String(30))
    start_offset_ms: Mapped[int] = mapped_column(BigInteger)
    end_offset_ms: Mapped[int] = mapped_column(BigInteger)
    peak_confidence: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))
    frame_count: Mapped[int] = mapped_column(Integer, default=1)
    evidence_id: Mapped[int | None] = mapped_column(ForeignKey("interview_evidence.evidence_id", ondelete="SET NULL"))
    details: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    evidence: Mapped[Evidence | None] = relationship()
