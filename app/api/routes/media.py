from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.api.deps import current_user
from app.core.errors import Forbidden, NotFound
from app.db.models import MediaFile, User
from app.db.session import get_db
from app.services.storage import get_storage

router = APIRouter(prefix="/api/media", tags=["Media"])


@router.get("/{file_id}", summary="A stored image (profile photo, ID capture) - owner only")
def get_media(file_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    media = db.get(MediaFile, file_id)
    if media is None:
        raise NotFound("File not found.")
    if media.owner_user_id != user.user_id:
        raise Forbidden("You do not have access to this file.")
    path = get_storage().path_for(media.storage_key)
    if not path.exists():
        raise NotFound("File not found.")
    return FileResponse(path, media_type=media.content_type, headers={"Cache-Control": "private, max-age=3600"})
