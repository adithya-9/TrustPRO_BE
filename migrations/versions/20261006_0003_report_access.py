"""Recruiter access to one candidate's report.

When a report completes, a recruiter login (random login ID + random password, user_type 'R')
is created for that report only and printed in the server terminal with the login link. The
login works for report_access.expires_at (7 days by default). Only the Argon2id hash of the
password is stored.

Deleting a report (or the candidate) removes its access rows; a trigger then removes the
recruiter login itself, so no orphan 'R' accounts are left behind.

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-06
"""
from __future__ import annotations

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

UTC_NOW = "(now() AT TIME ZONE 'utc')"

UPGRADE_SQL = f"""
CREATE TABLE epsoft.report_access (
    access_id BIGINT NOT NULL GENERATED ALWAYS AS IDENTITY
        (SEQUENCE NAME epsoft.report_access_access_id_seq) PRIMARY KEY,
    report_id BIGINT NOT NULL REFERENCES epsoft.interview_reports (report_id) ON DELETE CASCADE,
    user_id BIGINT NOT NULL REFERENCES epsoft.users (user_id) ON DELETE CASCADE,
    expires_at TIMESTAMP NOT NULL,
    last_login_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT {UTC_NOW},
    CONSTRAINT uq_report_access_user UNIQUE (user_id)
);
CREATE INDEX ix_report_access_report ON epsoft.report_access (report_id);
COMMENT ON TABLE epsoft.report_access IS 'Recruiter login (users.user_type = R) that can open exactly one report';
COMMENT ON COLUMN epsoft.report_access.expires_at IS 'UTC; the login and its sessions stop working after this';

CREATE OR REPLACE FUNCTION epsoft.fn_report_access_drop_login() RETURNS trigger AS $$
BEGIN
    DELETE FROM epsoft.users WHERE user_id = OLD.user_id AND user_type = 'R';
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_report_access_drop_login
    AFTER DELETE ON epsoft.report_access
    FOR EACH ROW EXECUTE FUNCTION epsoft.fn_report_access_drop_login();
"""

DOWNGRADE_SQL = """
DELETE FROM epsoft.users WHERE user_id IN (SELECT user_id FROM epsoft.report_access);
DROP TABLE IF EXISTS epsoft.report_access;
DROP FUNCTION IF EXISTS epsoft.fn_report_access_drop_login();
"""


def upgrade() -> None:
    op.execute(UPGRADE_SQL)


def downgrade() -> None:
    op.execute(DOWNGRADE_SQL)
