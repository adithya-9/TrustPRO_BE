"""Profile = first/last name, DOB, mobile, city, photo. ID check moves to report time.

- candidates: full_name -> first_name + last_name; add mobile_number, city.
  The government-ID fields are removed from the profile (ID type, ID number, uploaded ID
  image and its portrait): the ID is now only captured on camera before the interview.
- id_verifications: a row is a camera capture ('C'); its analysis (OCR name/DOB match,
  ID photo vs profile photo) is written during report generation ('A').
- interviews: gaze_calibration keeps the candidate's neutral gaze measured during the
  pre-interview countdown, so a reconnecting live session can reuse it.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-03
"""
from __future__ import annotations

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

UPGRADE_SQL = r"""
-- ------------------------------------------------------------------ candidates
ALTER TABLE epsoft.candidates DROP CONSTRAINT IF EXISTS ck_candidates_complete;

ALTER TABLE epsoft.candidates
    ADD COLUMN first_name VARCHAR(75),
    ADD COLUMN last_name VARCHAR(75),
    ADD COLUMN mobile_number VARCHAR(20),
    ADD COLUMN city VARCHAR(100);

-- Best-effort split of existing names: last word = last name, the rest = first name.
UPDATE epsoft.candidates
   SET first_name = CASE WHEN strpos(btrim(full_name), ' ') > 0
                         THEN regexp_replace(btrim(full_name), '\s+\S+$', '')
                         ELSE btrim(full_name) END,
       last_name  = CASE WHEN strpos(btrim(full_name), ' ') > 0
                         THEN substring(btrim(full_name) from '(\S+)$')
                         ELSE NULL END
 WHERE full_name IS NOT NULL;
UPDATE epsoft.candidates SET profile_status = 'I'
 WHERE profile_status = 'C' AND (first_name IS NULL OR last_name IS NULL);

ALTER TABLE epsoft.candidates
    DROP COLUMN full_name,
    DROP COLUMN id_document_type,
    DROP COLUMN id_number_hash,
    DROP COLUMN id_number_last4,
    DROP COLUMN id_document_file_id,
    DROP COLUMN id_portrait_file_id;

ALTER TABLE epsoft.candidates ADD CONSTRAINT ck_candidates_complete CHECK (
    profile_status = 'I' OR (first_name IS NOT NULL AND last_name IS NOT NULL
        AND date_of_birth IS NOT NULL AND profile_photo_file_id IS NOT NULL)
);
COMMENT ON COLUMN epsoft.candidates.first_name IS 'As printed on the government ID (compared at report time, order-insensitive)';
COMMENT ON COLUMN epsoft.candidates.mobile_number IS 'Contact number; not used for identity matching';

-- ------------------------------------------------------------------ id_verifications
ALTER TABLE epsoft.id_verifications DROP CONSTRAINT IF EXISTS id_verifications_status_check;
UPDATE epsoft.id_verifications SET status = 'A';
ALTER TABLE epsoft.id_verifications
    ADD CONSTRAINT ck_id_verifications_status CHECK (status IN ('C', 'A')),
    DROP COLUMN id_number_match,
    DROP COLUMN face_similarity_id_document,
    ADD COLUMN name_matched BOOLEAN,
    ADD COLUMN analysed_at TIMESTAMP;
COMMENT ON COLUMN epsoft.id_verifications.status IS
    'C = captured on camera (not analysed yet), A = analysed during report generation';

-- ------------------------------------------------------------------ interviews
ALTER TABLE epsoft.interviews ADD COLUMN gaze_calibration JSONB;
COMMENT ON COLUMN epsoft.interviews.gaze_calibration IS
    'Neutral gaze measured during the pre-interview countdown (medians and noise)';
"""

DOWNGRADE_SQL = r"""
ALTER TABLE epsoft.interviews DROP COLUMN IF EXISTS gaze_calibration;

ALTER TABLE epsoft.id_verifications
    DROP CONSTRAINT IF EXISTS ck_id_verifications_status,
    DROP COLUMN IF EXISTS name_matched,
    DROP COLUMN IF EXISTS analysed_at,
    ADD COLUMN id_number_match BOOLEAN,
    ADD COLUMN face_similarity_id_document NUMERIC(6, 4);
UPDATE epsoft.id_verifications SET status = 'M';
ALTER TABLE epsoft.id_verifications
    ADD CONSTRAINT id_verifications_status_check CHECK (status IN ('P', 'R', 'M'));

ALTER TABLE epsoft.candidates DROP CONSTRAINT IF EXISTS ck_candidates_complete;
ALTER TABLE epsoft.candidates
    ADD COLUMN full_name VARCHAR(150),
    ADD COLUMN id_document_type CHAR(1) CHECK (id_document_type IN ('A', 'P', 'D', 'S', 'V', 'O')),
    ADD COLUMN id_number_hash CHAR(64),
    ADD COLUMN id_number_last4 VARCHAR(4),
    ADD COLUMN id_document_file_id BIGINT REFERENCES epsoft.media_files (file_id) ON DELETE SET NULL,
    ADD COLUMN id_portrait_file_id BIGINT REFERENCES epsoft.media_files (file_id) ON DELETE SET NULL;
UPDATE epsoft.candidates SET full_name = btrim(coalesce(first_name, '') || ' ' || coalesce(last_name, '')),
                             profile_status = 'I';
ALTER TABLE epsoft.candidates
    DROP COLUMN first_name, DROP COLUMN last_name, DROP COLUMN mobile_number, DROP COLUMN city;
ALTER TABLE epsoft.candidates ADD CONSTRAINT ck_candidates_complete CHECK (
    profile_status = 'I' OR (full_name IS NOT NULL AND date_of_birth IS NOT NULL
        AND id_document_type IS NOT NULL AND profile_photo_file_id IS NOT NULL
        AND id_document_file_id IS NOT NULL)
);
"""


def upgrade() -> None:
    op.execute(UPGRADE_SQL)


def downgrade() -> None:
    op.execute(DOWNGRADE_SQL)
