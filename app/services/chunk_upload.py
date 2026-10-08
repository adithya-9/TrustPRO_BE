"""Fast path for recording-chunk uploads.

The browser uploads a recording part every few seconds for the whole assessment. With a remote
database (Neon), every SQL round trip costs ~70-300 ms, and the generic request path needed ~9
round trips per part (session, user, interview, candidate, recording, two updates, begin,
commit): seconds per part, so uploads fell behind the recording.

Here, per part:
  * the signed-in user comes from a short-lived in-memory session cache (logout clears it),
  * the interview's owner, storage key and next expected part come from an in-memory upload
    state, loaded from the database once per assessment,
  * one autocommit SQL statement records the new part count and file size.
The rules are the same as interview_service.append_chunk: parts in order, retries acknowledged,
size limits, WebM check on part 0. If the database disagrees with the cached state (e.g. another
server process), the state is reloaded and the browser is told which part to send next.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from sqlalchemy import text

from app.core import security
from app.core.config import get_settings
from app.core.errors import Conflict, Forbidden, InvalidUpload, NotAuthenticated, NotFound
from app.db.models import InterviewStatus, RecordingStatus, UserType, utcnow
from app.db.session import engine
from app.services.storage import get_storage

SESSION_TTL_S = 120
WEBM_MAGIC = b"\x1a\x45\xdf\xa3"


# ------------------------------------------------------------------ session cache
@dataclass
class _CachedSession:
    user_id: int
    user_type: str
    expires_at: object      # naive UTC datetime
    cached_at: float


_sessions: dict[str, _CachedSession] = {}
_sessions_lock = threading.Lock()


def forget_session(token: str | None) -> None:
    if token:
        with _sessions_lock:
            _sessions.pop(security.hash_token(token), None)


def candidate_id_for(token: str | None) -> int:
    """User id of the signed-in candidate, from cache when fresh (one query on a miss)."""
    if not token:
        raise NotAuthenticated("Please sign in to continue.")
    key = security.hash_token(token)
    now = time.monotonic()
    with _sessions_lock:
        cached = _sessions.get(key)
    if cached is None or now - cached.cached_at > SESSION_TTL_S or cached.expires_at <= utcnow():
        schema = get_settings().db_schema
        with engine.connect() as conn:
            row = conn.execute(text(
                f"SELECT s.user_id, s.expires_at, s.revoked_at, u.user_type, u.is_disabled "
                f"FROM {schema}.user_sessions s JOIN {schema}.users u ON u.user_id = s.user_id "
                f"WHERE s.token_hash = :h"), {"h": key}).first()
        if row is None or row.revoked_at is not None or row.expires_at <= utcnow() or row.is_disabled:
            forget_session(token)
            raise NotAuthenticated("Your session has expired. Please sign in again.", code="SESSION_EXPIRED")
        cached = _CachedSession(row.user_id, row.user_type, row.expires_at, now)
        with _sessions_lock:
            _sessions[key] = cached
    if cached.user_type != UserType.CANDIDATE:
        raise NotAuthenticated("Please sign in to continue.")
    return cached.user_id


# ------------------------------------------------------------------ upload state
@dataclass
class _UploadState:
    owner_user_id: int
    storage_key: str
    next_seq: int
    size: int
    status: str
    lock: threading.Lock


_uploads: dict[int, _UploadState] = {}
_uploads_lock = threading.Lock()


def forget_upload(interview_id: int) -> None:
    with _uploads_lock:
        _uploads.pop(interview_id, None)


def _load_state(interview_id: int) -> _UploadState:
    schema = get_settings().db_schema
    with engine.connect() as conn:
        row = conn.execute(text(
            f"SELECT c.user_id, i.status, i.recording_status, i.recording_chunks, m.storage_key, m.size_bytes "
            f"FROM {schema}.interviews i JOIN {schema}.candidates c ON c.candidate_id = i.candidate_id "
            f"LEFT JOIN {schema}.media_files m ON m.file_id = i.recording_file_id "
            f"WHERE i.interview_id = :iid"), {"iid": interview_id}).first()
    if row is None:
        raise NotFound("Interview not found.")
    if row.status not in (InterviewStatus.LIVE, InterviewStatus.ENDED) or row.storage_key is None:
        raise Conflict("The interview is not recording.", code="NOT_RECORDING")
    if row.recording_status == RecordingStatus.COMPLETE:
        raise Conflict("The recording is already complete.", code="RECORDING_COMPLETE")
    return _UploadState(row.user_id, row.storage_key, row.recording_chunks, row.size_bytes or 0, row.status,
                        threading.Lock())


def append(interview_id: int, user_id: int, seq: int, data: bytes) -> tuple[int, int]:
    """(next_seq, bytes stored) after storing part `seq`."""
    settings = get_settings()
    with _uploads_lock:
        state = _uploads.get(interview_id)
    if state is None:
        state = _load_state(interview_id)
        with _uploads_lock:
            state = _uploads.setdefault(interview_id, state)
    if state.owner_user_id != user_id:
        raise Forbidden("You do not have access to this interview.")
    with state.lock:
        if seq < state.next_seq:                       # retry of a stored part: acknowledge it
            return state.next_seq, state.size
        if seq > state.next_seq:
            raise Conflict("A part of the recording is missing.", code="CHUNK_OUT_OF_ORDER",
                           details={"expected_seq": state.next_seq})
        if not data:
            raise InvalidUpload("Empty recording chunk.", code="EMPTY_CHUNK")
        if len(data) > settings.max_video_chunk_bytes:
            raise InvalidUpload("Recording chunk is too large.", code="CHUNK_TOO_LARGE")
        if seq == 0 and not data.startswith(WEBM_MAGIC):
            raise InvalidUpload("The recording must be a WebM video.", code="INVALID_VIDEO")
        if state.size + len(data) > settings.max_video_bytes:
            raise InvalidUpload("The recording is larger than the allowed maximum.", code="RECORDING_TOO_LARGE")
        size = get_storage().append(state.storage_key, data)
        schema = settings.db_schema
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            updated = conn.execute(text(
                f"WITH iv AS (UPDATE {schema}.interviews SET recording_chunks = :n, updated_at = :now "
                f"WHERE interview_id = :iid AND recording_chunks = :seq RETURNING recording_file_id) "
                f"UPDATE {schema}.media_files m SET size_bytes = :size FROM iv "
                f"WHERE m.file_id = iv.recording_file_id RETURNING m.file_id"),
                {"n": seq + 1, "now": utcnow(), "iid": interview_id, "seq": seq, "size": size}).first()
        if updated is None:                            # the database moved on without us: resync
            forget_upload(interview_id)
            fresh = _load_state(interview_id)
            raise Conflict("A part of the recording is missing.", code="CHUNK_OUT_OF_ORDER",
                           details={"expected_seq": fresh.next_seq})
        state.next_seq, state.size = seq + 1, size
        return state.next_seq, state.size


def is_live(interview_id: int) -> bool:
    with _uploads_lock:
        state = _uploads.get(interview_id)
    return state is not None and state.status == InterviewStatus.LIVE
