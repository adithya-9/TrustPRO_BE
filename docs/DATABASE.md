# TrustPRO database design

## What the inspection found (2026-10-01)

`trustmate` (PostgreSQL 17.10) was inspected before any change:

- Schemas: `epsoft` (owner postgres) and `public`.
- **`epsoft` contained no tables, views, sequences, types or functions.** `public` was also
  empty. Only the `plpgsql` extension was installed.
- There were therefore no existing Trustmate candidate, user or interview tables to reuse.

Because the database gave no conventions to follow, the conventions were taken from the
sibling EPSoft products that already use an `epsoft` schema (TubeLadder recruiter and
job-seeker: `tl-recruiter/DBChanges/*.sql` and their JPA entities):

| Convention observed | Applied in TrustPRO |
|---|---|
| Plural snake_case tables (`recruiters`, `job_details`) | `candidates`, `interviews`, `interview_gaze_events`, ... |
| `<entity>_id BIGINT GENERATED ALWAYS AS IDENTITY (SEQUENCE NAME <table>_<col>_seq)` | Same for every table |
| `fk_<table>_<column>` / `idx_<table>_<column>` names, `ON DELETE CASCADE` to owners | Same |
| Shared `epsoft.users` (email, password_hash, user_type CHAR(1), failed_attempts, is_disabled, last_active) referenced by product tables | Created with the same shape, so other Trustmate products can reuse it |
| `created_at` / `updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP` | Same columns; values are UTC (`now() AT TIME ZONE 'utc'`) |
| One-letter status codes with `CHECK` constraints | Used for every *status*; codes documented with `COMMENT ON COLUMN` |

Two deliberate deviations:
- Detection *taxonomies* (gaze direction, environment event type) use short readable
  `VARCHAR` codes (`LEFT`, `PHONE`) with CHECK constraints rather than single letters,
  because they appear directly in reports and evidence queries.
- The sibling scripts also indexed primary keys, which is redundant in PostgreSQL. TrustPRO
  does not.

No separate schema was created. TrustPRO's Alembic version table is
`epsoft.trustpro_alembic_version`, so other products can run their own Alembic history in
the same schema.

## Tables

```
users 1-1 candidates 1-n id_verifications
  |           |                 |
  |           +-1-n interviews -+ (interview admitted by one verification)
  |                   |-1-n interview_gaze_events      (live, one row per gaze segment)
  |                   |-1-n interview_reports 1-n interview_evidence
  |                   |-1-n interview_environment_events (live 'L' or analysis 'A') -> evidence
  +-1-n media_files  (metadata for every stored binary; bytes live in file storage)
  +-1-n user_sessions
```

| Table | Purpose | Notable columns / constraints |
|---|---|---|
| `users` | Shared account | `email` unique and lower-case (CHECK), Argon2id `password_hash`, `user_type` C/R/A, lockout counter |
| `user_sessions` | Opaque sessions (no JWT) | only the SHA-256 of the cookie token; `expires_at`, `revoked_at` |
| `media_files` | Binary metadata | `file_category` P/I/C/T/V/E, internal `storage_key` (never sent to clients), size, SHA-256 |
| `candidates` | Profile | first / last name, DOB, mobile, city, profile photo; CHECK that a complete profile has the required fields |
| `id_verifications` | Each camera capture of the government ID | C/A status (captured / analysed at report time), name / DOB / photo results, quality JSON, OCR fields |
| `interviews` | Lifecycle | C/L/E/A status, admitted-by `verification_id`, recording reference and N/U/C/F upload status, chunk counter |
| `interview_gaze_events` | Gaze segments | direction, start/end offset (ms from recording start), **generated** `duration_ms`, average confidence, frame count |
| `interview_reports` | Report job + results | Q/P/C/F status, progress, stage, error, sample interval, JSON summaries per section, processing info |
| `interview_evidence` | Frame + model result | kind, label, offset, image file, confidence, `model_result` JSONB; `report_id` NULL for live evidence |
| `interview_environment_events` | Environment events | source L/A, type, offsets, peak confidence, frames, `evidence_id`; CHECK that analysis events belong to a report |

Large binaries are **not** stored in PostgreSQL. Recordings and images go to the storage
folder (`STORAGE_DIR`), which can be swapped for object storage (`services/storage.py`).
Per-frame data is kept out of the database; only segments, events and selected evidence are
stored.

## Migration 0002 (2026-10-03): profile and ID check rework

- `candidates`:
  - `full_name` is replaced by `first_name` + `last_name`. Existing names were split as
    "last word = last name".
  - `mobile_number` and `city` are added.
  - The government-ID fields are removed from the profile: `id_document_type`, `id_number_hash`,
    `id_number_last4`, `id_document_file_id` and `id_portrait_file_id`. The ID is now only shown
    on camera before the interview.
  - The completeness CHECK now requires first name, last name, date of birth and profile photo.
- `id_verifications`:
  - A row is a camera capture with status `C`. Report generation analyses it and sets status `A`,
    writing `name_matched`, `name_match_score`, `dob_match`, `face_similarity_profile`,
    `extracted_fields` (including the matched ID text and extra words) and `analysed_at`.
  - The ID-number and uploaded-ID-photo columns are removed.
- `interviews.gaze_calibration` (JSONB) stores the candidate's neutral gaze measured during the
  pre-interview countdown, so a reconnecting live session reuses it.

A `pg_dump` of the schema was taken before this migration was applied.

## Indexes

Only lookups the application performs:
- `idx_user_sessions_user_id`, `idx_media_files_owner_user_id`
- `idx_id_verifications_candidate_id` (latest verification per candidate)
- `idx_interviews_candidate_id` (dashboard)
- `idx_interview_gaze_events_interview_id (interview_id, start_offset_ms)` (timeline in order)
- `idx_interview_reports_interview_id`
- `idx_interview_evidence_report_id`, `idx_interview_evidence_interview_id`
- `idx_interview_environment_events_interview_id (interview_id, start_offset_ms)`
- `idx_interview_environment_events_report_id` **partial** (`WHERE report_id IS NOT NULL`)

Unique constraints (`users.email`, `user_sessions.token_hash`, `media_files.storage_key`,
`candidates.user_id`) provide their own indexes.

## Migrations

```
alembic upgrade head      # create / update
alembic downgrade base    # remove all TrustPRO tables (development only)
alembic upgrade head --sql > trustpro.sql   # offline SQL for a DBA review
```

The migration (`migrations/versions/20261001_0001_trustpro_initial.py`) is plain SQL so it
reads like the existing EPSoft `DBChanges` scripts and can be reviewed by a DBA.
