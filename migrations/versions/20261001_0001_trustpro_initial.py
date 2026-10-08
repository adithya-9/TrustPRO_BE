"""TrustPRO initial schema in trustmate.epsoft.

Conventions follow the existing EPSoft products that use the same `epsoft` schema
(TubeLadder recruiter / job-seeker): plural snake_case tables, BIGINT identity keys with
named sequences, fk_<table>_<column> / idx_<table>_<column> names, created_at/updated_at
TIMESTAMP columns, one-letter status codes guarded by CHECK constraints, and a shared
epsoft.users account table that product tables reference.

Timestamps are stored as UTC.

Revision ID: 0001
Revises:
Create Date: 2026-10-01
"""
from __future__ import annotations

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

UTC_NOW = "(now() AT TIME ZONE 'utc')"

UPGRADE_SQL = f"""
-- ------------------------------------------------------------------ users
-- Shared account table (same shape as epsoft.users in other EPSoft products).
CREATE TABLE IF NOT EXISTS epsoft.users (
    user_id BIGINT NOT NULL GENERATED ALWAYS AS IDENTITY
        (SEQUENCE NAME epsoft.users_user_id_seq) PRIMARY KEY,
    email VARCHAR(255) NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    user_type CHAR(1) NOT NULL DEFAULT 'C' CHECK (user_type IN ('C', 'R', 'A')),
    failed_attempts INT NOT NULL DEFAULT 0,
    is_disabled BOOLEAN NOT NULL DEFAULT FALSE,
    last_active TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT {UTC_NOW},
    updated_at TIMESTAMP NOT NULL DEFAULT {UTC_NOW},
    CONSTRAINT uq_users_email UNIQUE (email),
    CONSTRAINT ck_users_email_lowercase CHECK (email = lower(email))
);
COMMENT ON COLUMN epsoft.users.user_type IS 'C = candidate, R = reviewer/recruiter, A = admin';
COMMENT ON COLUMN epsoft.users.password_hash IS 'Argon2id hash (PHC string). Plaintext is never stored.';

-- ------------------------------------------------------------------ user_sessions
-- Opaque server-side sessions (no JWT). Only a SHA-256 of the cookie token is stored.
CREATE TABLE IF NOT EXISTS epsoft.user_sessions (
    session_id BIGINT NOT NULL GENERATED ALWAYS AS IDENTITY
        (SEQUENCE NAME epsoft.user_sessions_session_id_seq) PRIMARY KEY,
    user_id BIGINT NOT NULL,
    token_hash CHAR(64) NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    revoked_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT {UTC_NOW},
    CONSTRAINT uq_user_sessions_token_hash UNIQUE (token_hash),
    CONSTRAINT fk_user_sessions_user_id FOREIGN KEY (user_id)
        REFERENCES epsoft.users (user_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON epsoft.user_sessions (user_id);

-- ------------------------------------------------------------------ media_files
-- Metadata for every stored binary (photos, ID images, recordings, evidence frames).
-- The bytes live in file/object storage; storage_key is internal and never sent to clients.
CREATE TABLE IF NOT EXISTS epsoft.media_files (
    file_id BIGINT NOT NULL GENERATED ALWAYS AS IDENTITY
        (SEQUENCE NAME epsoft.media_files_file_id_seq) PRIMARY KEY,
    owner_user_id BIGINT NOT NULL,
    file_category CHAR(1) NOT NULL CHECK (file_category IN ('P', 'I', 'C', 'T', 'V', 'E')),
    storage_key VARCHAR(255) NOT NULL,
    content_type VARCHAR(100) NOT NULL,
    size_bytes BIGINT NOT NULL DEFAULT 0 CHECK (size_bytes >= 0),
    sha256 CHAR(64),
    width INT,
    height INT,
    created_at TIMESTAMP NOT NULL DEFAULT {UTC_NOW},
    CONSTRAINT uq_media_files_storage_key UNIQUE (storage_key),
    CONSTRAINT fk_media_files_owner_user_id FOREIGN KEY (owner_user_id)
        REFERENCES epsoft.users (user_id) ON DELETE CASCADE
);
COMMENT ON COLUMN epsoft.media_files.file_category IS
    'P = profile photo, I = uploaded ID document, C = ID camera capture, T = ID portrait crop, V = interview recording, E = evidence frame';
CREATE INDEX IF NOT EXISTS idx_media_files_owner_user_id ON epsoft.media_files (owner_user_id);

-- ------------------------------------------------------------------ candidates
CREATE TABLE IF NOT EXISTS epsoft.candidates (
    candidate_id BIGINT NOT NULL GENERATED ALWAYS AS IDENTITY
        (SEQUENCE NAME epsoft.candidates_candidate_id_seq) PRIMARY KEY,
    user_id BIGINT NOT NULL,
    full_name VARCHAR(150),
    date_of_birth DATE,
    id_document_type CHAR(1) CHECK (id_document_type IN ('A', 'P', 'D', 'S', 'V', 'O')),
    id_number_hash CHAR(64),
    id_number_last4 VARCHAR(4),
    profile_photo_file_id BIGINT,
    id_document_file_id BIGINT,
    id_portrait_file_id BIGINT,
    profile_status CHAR(1) NOT NULL DEFAULT 'I' CHECK (profile_status IN ('I', 'C')),
    created_at TIMESTAMP NOT NULL DEFAULT {UTC_NOW},
    updated_at TIMESTAMP NOT NULL DEFAULT {UTC_NOW},
    CONSTRAINT uq_candidates_user_id UNIQUE (user_id),
    CONSTRAINT fk_candidates_user_id FOREIGN KEY (user_id)
        REFERENCES epsoft.users (user_id) ON DELETE CASCADE,
    CONSTRAINT fk_candidates_profile_photo_file_id FOREIGN KEY (profile_photo_file_id)
        REFERENCES epsoft.media_files (file_id) ON DELETE SET NULL,
    CONSTRAINT fk_candidates_id_document_file_id FOREIGN KEY (id_document_file_id)
        REFERENCES epsoft.media_files (file_id) ON DELETE SET NULL,
    CONSTRAINT fk_candidates_id_portrait_file_id FOREIGN KEY (id_portrait_file_id)
        REFERENCES epsoft.media_files (file_id) ON DELETE SET NULL,
    CONSTRAINT ck_candidates_complete CHECK (
        profile_status = 'I' OR (full_name IS NOT NULL AND date_of_birth IS NOT NULL
            AND id_document_type IS NOT NULL AND profile_photo_file_id IS NOT NULL
            AND id_document_file_id IS NOT NULL)
    )
);
COMMENT ON COLUMN epsoft.candidates.id_document_type IS
    'A = Aadhaar, P = PAN card, D = driving licence, S = passport, V = voter ID, O = other government ID';
COMMENT ON COLUMN epsoft.candidates.id_number_hash IS
    'HMAC-SHA256 of the normalised ID number (keyed with ID_NUMBER_PEPPER). The full number is not stored.';
COMMENT ON COLUMN epsoft.candidates.profile_status IS 'I = incomplete, C = complete';

-- ------------------------------------------------------------------ id_verifications
-- One row per live ID capture attempt before an interview.
CREATE TABLE IF NOT EXISTS epsoft.id_verifications (
    verification_id BIGINT NOT NULL GENERATED ALWAYS AS IDENTITY
        (SEQUENCE NAME epsoft.id_verifications_verification_id_seq) PRIMARY KEY,
    candidate_id BIGINT NOT NULL,
    attempt_no INT NOT NULL CHECK (attempt_no > 0),
    status CHAR(1) NOT NULL CHECK (status IN ('P', 'R', 'M')),
    capture_file_id BIGINT,
    portrait_file_id BIGINT,
    portrait_detected BOOLEAN NOT NULL DEFAULT FALSE,
    text_readable BOOLEAN NOT NULL DEFAULT FALSE,
    name_match_score NUMERIC(5, 2),
    dob_match BOOLEAN,
    id_number_match BOOLEAN,
    face_similarity_profile NUMERIC(6, 4),
    face_similarity_id_document NUMERIC(6, 4),
    quality JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    extracted_fields JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    issues JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMP NOT NULL DEFAULT {UTC_NOW},
    CONSTRAINT fk_id_verifications_candidate_id FOREIGN KEY (candidate_id)
        REFERENCES epsoft.candidates (candidate_id) ON DELETE CASCADE,
    CONSTRAINT fk_id_verifications_capture_file_id FOREIGN KEY (capture_file_id)
        REFERENCES epsoft.media_files (file_id) ON DELETE SET NULL,
    CONSTRAINT fk_id_verifications_portrait_file_id FOREIGN KEY (portrait_file_id)
        REFERENCES epsoft.media_files (file_id) ON DELETE SET NULL
);
COMMENT ON COLUMN epsoft.id_verifications.status IS
    'P = passed, R = retake required, M = accepted for manual review (details did not fully match after repeated attempts)';
COMMENT ON COLUMN epsoft.id_verifications.extracted_fields IS
    'OCR fields read from the ID. ID numbers are stored masked.';
CREATE INDEX IF NOT EXISTS idx_id_verifications_candidate_id ON epsoft.id_verifications (candidate_id);

-- ------------------------------------------------------------------ interviews
CREATE TABLE IF NOT EXISTS epsoft.interviews (
    interview_id BIGINT NOT NULL GENERATED ALWAYS AS IDENTITY
        (SEQUENCE NAME epsoft.interviews_interview_id_seq) PRIMARY KEY,
    candidate_id BIGINT NOT NULL,
    verification_id BIGINT NOT NULL,
    status CHAR(1) NOT NULL DEFAULT 'C' CHECK (status IN ('C', 'L', 'E', 'A')),
    started_at TIMESTAMP,
    ended_at TIMESTAMP,
    duration_ms BIGINT CHECK (duration_ms >= 0),
    recording_file_id BIGINT,
    recording_status CHAR(1) NOT NULL DEFAULT 'N' CHECK (recording_status IN ('N', 'U', 'C', 'F')),
    recording_chunks INT NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT {UTC_NOW},
    updated_at TIMESTAMP NOT NULL DEFAULT {UTC_NOW},
    CONSTRAINT fk_interviews_candidate_id FOREIGN KEY (candidate_id)
        REFERENCES epsoft.candidates (candidate_id) ON DELETE CASCADE,
    CONSTRAINT fk_interviews_verification_id FOREIGN KEY (verification_id)
        REFERENCES epsoft.id_verifications (verification_id),
    CONSTRAINT fk_interviews_recording_file_id FOREIGN KEY (recording_file_id)
        REFERENCES epsoft.media_files (file_id) ON DELETE SET NULL
);
COMMENT ON COLUMN epsoft.interviews.status IS 'C = created, L = live, E = ended, A = abandoned';
COMMENT ON COLUMN epsoft.interviews.recording_status IS 'N = not started, U = uploading, C = complete, F = failed';
CREATE INDEX IF NOT EXISTS idx_interviews_candidate_id ON epsoft.interviews (candidate_id);

-- ------------------------------------------------------------------ interview_gaze_events
-- Stable gaze segments from the live monitor (one row per segment, not per frame).
-- Offsets are milliseconds from the start of the recording, so they line up with the video.
CREATE TABLE IF NOT EXISTS epsoft.interview_gaze_events (
    gaze_event_id BIGINT NOT NULL GENERATED ALWAYS AS IDENTITY
        (SEQUENCE NAME epsoft.interview_gaze_events_gaze_event_id_seq) PRIMARY KEY,
    interview_id BIGINT NOT NULL,
    direction VARCHAR(10) NOT NULL CHECK (direction IN ('FORWARD', 'LEFT', 'RIGHT', 'UP', 'DOWN', 'NO_FACE')),
    start_offset_ms BIGINT NOT NULL CHECK (start_offset_ms >= 0),
    end_offset_ms BIGINT NOT NULL,
    duration_ms BIGINT GENERATED ALWAYS AS (end_offset_ms - start_offset_ms) STORED,
    avg_confidence NUMERIC(4, 3),
    frame_count INT NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT {UTC_NOW},
    CONSTRAINT ck_interview_gaze_events_range CHECK (end_offset_ms >= start_offset_ms),
    CONSTRAINT fk_interview_gaze_events_interview_id FOREIGN KEY (interview_id)
        REFERENCES epsoft.interviews (interview_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_interview_gaze_events_interview_id
    ON epsoft.interview_gaze_events (interview_id, start_offset_ms);

-- ------------------------------------------------------------------ interview_reports
CREATE TABLE IF NOT EXISTS epsoft.interview_reports (
    report_id BIGINT NOT NULL GENERATED ALWAYS AS IDENTITY
        (SEQUENCE NAME epsoft.interview_reports_report_id_seq) PRIMARY KEY,
    interview_id BIGINT NOT NULL,
    status CHAR(1) NOT NULL DEFAULT 'Q' CHECK (status IN ('Q', 'P', 'C', 'F')),
    progress_pct SMALLINT NOT NULL DEFAULT 0 CHECK (progress_pct BETWEEN 0 AND 100),
    current_stage VARCHAR(60),
    error_message VARCHAR(500),
    sample_interval_ms INT NOT NULL,
    frames_analyzed INT NOT NULL DEFAULT 0,
    identity_summary JSONB,
    environment_summary JSONB,
    gaze_summary JSONB,
    processing_info JSONB,
    requested_at TIMESTAMP NOT NULL DEFAULT {UTC_NOW},
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT {UTC_NOW},
    updated_at TIMESTAMP NOT NULL DEFAULT {UTC_NOW},
    CONSTRAINT fk_interview_reports_interview_id FOREIGN KEY (interview_id)
        REFERENCES epsoft.interviews (interview_id) ON DELETE CASCADE
);
COMMENT ON COLUMN epsoft.interview_reports.status IS 'Q = queued, P = processing, C = completed, F = failed';
CREATE INDEX IF NOT EXISTS idx_interview_reports_interview_id ON epsoft.interview_reports (interview_id);

-- ------------------------------------------------------------------ interview_evidence
-- A captured frame plus the exact model result that made it evidence.
CREATE TABLE IF NOT EXISTS epsoft.interview_evidence (
    evidence_id BIGINT NOT NULL GENERATED ALWAYS AS IDENTITY
        (SEQUENCE NAME epsoft.interview_evidence_evidence_id_seq) PRIMARY KEY,
    interview_id BIGINT NOT NULL,
    report_id BIGINT,
    evidence_kind VARCHAR(30) NOT NULL CHECK (evidence_kind IN ('IDENTITY', 'ENVIRONMENT')),
    label VARCHAR(40) NOT NULL,
    offset_ms BIGINT NOT NULL CHECK (offset_ms >= 0),
    file_id BIGINT,
    confidence NUMERIC(5, 4),
    model_result JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    created_at TIMESTAMP NOT NULL DEFAULT {UTC_NOW},
    CONSTRAINT fk_interview_evidence_interview_id FOREIGN KEY (interview_id)
        REFERENCES epsoft.interviews (interview_id) ON DELETE CASCADE,
    CONSTRAINT fk_interview_evidence_report_id FOREIGN KEY (report_id)
        REFERENCES epsoft.interview_reports (report_id) ON DELETE CASCADE,
    CONSTRAINT fk_interview_evidence_file_id FOREIGN KEY (file_id)
        REFERENCES epsoft.media_files (file_id) ON DELETE SET NULL
);
COMMENT ON COLUMN epsoft.interview_evidence.report_id IS 'NULL for evidence captured live during the interview';
CREATE INDEX IF NOT EXISTS idx_interview_evidence_report_id ON epsoft.interview_evidence (report_id);
CREATE INDEX IF NOT EXISTS idx_interview_evidence_interview_id ON epsoft.interview_evidence (interview_id);

-- ------------------------------------------------------------------ interview_environment_events
CREATE TABLE IF NOT EXISTS epsoft.interview_environment_events (
    environment_event_id BIGINT NOT NULL GENERATED ALWAYS AS IDENTITY
        (SEQUENCE NAME epsoft.interview_environment_events_environment_event_id_seq) PRIMARY KEY,
    interview_id BIGINT NOT NULL,
    report_id BIGINT,
    source CHAR(1) NOT NULL CHECK (source IN ('L', 'A')),
    event_type VARCHAR(30) NOT NULL CHECK (event_type IN
        ('NO_PERSON', 'ADDITIONAL_PERSON', 'PHONE', 'LAPTOP', 'SCREEN', 'BOOK')),
    start_offset_ms BIGINT NOT NULL CHECK (start_offset_ms >= 0),
    end_offset_ms BIGINT NOT NULL,
    peak_confidence NUMERIC(5, 4),
    frame_count INT NOT NULL DEFAULT 1,
    evidence_id BIGINT,
    details JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    created_at TIMESTAMP NOT NULL DEFAULT {UTC_NOW},
    CONSTRAINT ck_interview_environment_events_range CHECK (end_offset_ms >= start_offset_ms),
    CONSTRAINT ck_interview_environment_events_report CHECK (source = 'L' OR report_id IS NOT NULL),
    CONSTRAINT fk_interview_environment_events_interview_id FOREIGN KEY (interview_id)
        REFERENCES epsoft.interviews (interview_id) ON DELETE CASCADE,
    CONSTRAINT fk_interview_environment_events_report_id FOREIGN KEY (report_id)
        REFERENCES epsoft.interview_reports (report_id) ON DELETE CASCADE,
    CONSTRAINT fk_interview_environment_events_evidence_id FOREIGN KEY (evidence_id)
        REFERENCES epsoft.interview_evidence (evidence_id) ON DELETE SET NULL
);
COMMENT ON COLUMN epsoft.interview_environment_events.source IS 'L = live monitor during the interview, A = post-interview analysis';
CREATE INDEX IF NOT EXISTS idx_interview_environment_events_interview_id
    ON epsoft.interview_environment_events (interview_id, start_offset_ms);
CREATE INDEX IF NOT EXISTS idx_interview_environment_events_report_id
    ON epsoft.interview_environment_events (report_id) WHERE report_id IS NOT NULL;
"""

DOWNGRADE_SQL = """
DROP TABLE IF EXISTS epsoft.interview_environment_events;
DROP TABLE IF EXISTS epsoft.interview_evidence;
DROP TABLE IF EXISTS epsoft.interview_reports;
DROP TABLE IF EXISTS epsoft.interview_gaze_events;
DROP TABLE IF EXISTS epsoft.interviews;
DROP TABLE IF EXISTS epsoft.id_verifications;
DROP TABLE IF EXISTS epsoft.candidates;
DROP TABLE IF EXISTS epsoft.media_files;
DROP TABLE IF EXISTS epsoft.user_sessions;
DROP TABLE IF EXISTS epsoft.users;
"""


def upgrade() -> None:
    op.execute(UPGRADE_SQL)


def downgrade() -> None:
    op.execute(DOWNGRADE_SQL)
