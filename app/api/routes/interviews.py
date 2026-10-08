from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.api.deps import current_user
from app.core.config import get_settings
from app.core.errors import InvalidUpload, NotFound
from app.db.models import RecordingStatus, User
from app.db.session import get_db
from app.schemas.interview import ChunkAck, EndInterviewRequest, InterviewOut
from app.services import interview_service
from app.services.storage import get_storage

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/interviews", tags=["Interviews"])


@router.get("", response_model=list[InterviewOut], summary="The candidate's interviews, newest first")
def list_interviews(user: User = Depends(current_user), db: Session = Depends(get_db)) -> list[InterviewOut]:
    return [interview_service.interview_out(db, i) for i in interview_service.list_interviews(db, user)]


@router.post("", response_model=InterviewOut, status_code=201, summary="Create an interview (requires ID verification)")
def create_interview(user: User = Depends(current_user), db: Session = Depends(get_db)) -> InterviewOut:
    return interview_service.interview_out(db, interview_service.create_interview(db, user))


@router.get("/{interview_id}", response_model=InterviewOut, summary="Interview details")
def get_interview(interview_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)) -> InterviewOut:
    return interview_service.interview_out(db, interview_service.get_owned_interview(db, user, interview_id))


@router.post("/{interview_id}/start", response_model=InterviewOut, summary="Start the interview and recording")
def start_interview(interview_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)) -> InterviewOut:
    interview = interview_service.get_owned_interview(db, user, interview_id)
    return interview_service.interview_out(db, interview_service.start_interview(db, interview))


@router.put("/{interview_id}/recording/chunks/{seq}", response_model=ChunkAck,
            summary="Append the next recording chunk (raw WebM bytes)")
async def upload_chunk(interview_id: int, seq: int, request: Request, user: User = Depends(current_user),
                       db: Session = Depends(get_db)) -> ChunkAck:
    if seq < 0:
        raise InvalidUpload("Invalid chunk number.")
    limit = get_settings().max_video_chunk_bytes
    data = bytearray()
    async for part in request.stream():
        data.extend(part)
        if len(data) > limit:
            raise InvalidUpload("Recording chunk is too large.", code="CHUNK_TOO_LARGE")

    def store() -> tuple[int, int]:
        interview = interview_service.get_owned_interview(db, user, interview_id)
        return interview_service.append_chunk(db, interview, seq, bytes(data))

    next_seq, size = await run_in_threadpool(store)
    return ChunkAck(next_seq=next_seq, bytes_received=size)


@router.post("/{interview_id}/end", response_model=InterviewOut, summary="End the assessment (the report is then generated on the server)")
def end_interview(interview_id: int, body: EndInterviewRequest, user: User = Depends(current_user),
                  db: Session = Depends(get_db)) -> InterviewOut:
    interview = interview_service.get_owned_interview(db, user, interview_id)
    return interview_service.interview_out(db, interview_service.end_interview(db, interview, body.duration_ms, body.chunk_count))


@router.get("/{interview_id}/recording", summary="Stream the interview recording (supports HTTP range requests)")
def get_recording(interview_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    interview = interview_service.get_owned_interview(db, user, interview_id)
    if interview.recording is None or interview.recording_status != RecordingStatus.COMPLETE:
        raise NotFound("The recording is not available.")
    path = get_storage().path_for(interview.recording.storage_key)
    return FileResponse(path, media_type="video/webm", headers={"Cache-Control": "private, max-age=3600"})
