from __future__ import annotations

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.errors import NotAuthenticated
from app.db.models import User, UserType
from app.db.session import get_db
from app.services import auth_service


def session_token(request: Request) -> str | None:
    return request.cookies.get(get_settings().session_cookie_name)


def current_user(request: Request, db: Session = Depends(get_db)) -> User:
    """The signed-in candidate. Recruiter logins only work on the /api/recruiter routes."""
    user = auth_service.resolve_session(db, session_token(request))
    if user.user_type != UserType.CANDIDATE:
        raise NotAuthenticated("Please sign in to continue.")
    return user
