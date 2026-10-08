from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy.orm import Session

from app.api.deps import current_user, session_token
from app.core.config import get_settings
from app.core.errors import NotAuthenticated
from app.db.models import ProfileStatus, User, UserType
from app.db.session import get_db
from app.schemas.auth import LoginRequest, RegisterRequest, UserOut
from app.schemas.common import MessageOut
from app.services import auth_service, candidate_service

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


def _set_cookie(response: Response, token: str) -> None:
    s = get_settings()
    response.set_cookie(
        s.session_cookie_name, token, max_age=s.session_ttl_hours * 3600,
        httponly=True, secure=s.cookie_secure, samesite="lax", path="/",
    )


def user_out(db: Session, user: User) -> UserOut:
    candidate = candidate_service.get_candidate(db, user)
    complete = candidate.profile_status == ProfileStatus.COMPLETE
    from app.services.verification_service import is_confirmed

    captured = complete and is_confirmed(candidate_service.latest_capture(db, candidate))   # ID details saved
    next_step = "PROFILE" if not complete else "INTERVIEW" if captured else "ID_VERIFICATION"
    return UserOut(user_id=user.user_id, email=user.email, user_type=user.user_type,
                   profile_complete=complete, id_captured=captured, next_step=next_step)


@router.post("/register", response_model=UserOut, status_code=201, summary="Create a candidate account and sign in")
def register(body: RegisterRequest, response: Response, db: Session = Depends(get_db)) -> UserOut:
    user = auth_service.register(db, body.email, body.password)
    _set_cookie(response, auth_service.create_session(db, user))
    return user_out(db, user)


@router.post("/login", response_model=UserOut, summary="Sign in with email and password")
def login(body: LoginRequest, response: Response, db: Session = Depends(get_db)) -> UserOut:
    user = auth_service.authenticate(db, body.email, body.password)
    if user.user_type != UserType.CANDIDATE:      # recruiter logins sign in at /api/recruiter/login
        raise NotAuthenticated("The email or password is incorrect.", code="INVALID_CREDENTIALS")
    _set_cookie(response, auth_service.create_session(db, user))
    return user_out(db, user)


@router.post("/logout", response_model=MessageOut, summary="Sign out and revoke the session")
def logout(request: Request, response: Response, db: Session = Depends(get_db)) -> MessageOut:
    auth_service.revoke_session(db, session_token(request))
    response.delete_cookie(get_settings().session_cookie_name, path="/")
    return MessageOut(message="Signed out.")


@router.get("/me", response_model=UserOut, summary="Current user and the next onboarding step")
def me(user: User = Depends(current_user), db: Session = Depends(get_db)) -> UserOut:
    return user_out(db, user)
