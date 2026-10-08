from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import ReportAccess, utcnow
from app.db.session import get_db
from app.schemas.auth import LoginRequest
from app.schemas.common import ApiModel, MessageOut, iso
from app.services import auth_service, recruiter_service

router = APIRouter(prefix="/api/recruiter", tags=["Recruiter"])

# The report is our own self-contained HTML (inline styles, data: images, no scripts).
REPORT_CSP = "default-src 'none'; img-src data:; style-src 'unsafe-inline'; font-src data:"


class RecruiterOut(ApiModel):
    login_id: str
    report_id: int
    candidate_name: str
    expires_at: str


def _token(request: Request) -> str | None:
    return request.cookies.get(get_settings().recruiter_cookie_name)


def current_access(request: Request, db: Session = Depends(get_db)) -> ReportAccess:
    return recruiter_service.resolve(db, _token(request))


def _out(access: ReportAccess) -> RecruiterOut:
    return RecruiterOut(login_id=access.user.email, report_id=access.report_id,
                        candidate_name=recruiter_service.candidate_name(access), expires_at=iso(access.expires_at))


@router.post("/login", response_model=RecruiterOut, summary="Sign in with the login ID and password printed for a report")
def login(body: LoginRequest, response: Response, db: Session = Depends(get_db)) -> RecruiterOut:
    token, access = recruiter_service.login(db, body.email, body.password)
    s = get_settings()
    max_age = max(60, int((access.expires_at - utcnow()).total_seconds()))
    response.set_cookie(s.recruiter_cookie_name, token, max_age=min(max_age, s.session_ttl_hours * 3600),
                        httponly=True, secure=s.cookie_secure, samesite="lax", path="/")
    return _out(access)


@router.get("/me", response_model=RecruiterOut, summary="The signed-in recruiter and the report they can open")
def me(access: ReportAccess = Depends(current_access)) -> RecruiterOut:
    return _out(access)


@router.get("/report", response_class=HTMLResponse, summary="The candidate report (HTML) for this recruiter login")
def report(access: ReportAccess = Depends(current_access), db: Session = Depends(get_db)) -> HTMLResponse:
    return HTMLResponse(recruiter_service.report_html(db, access),
                        headers={"Cache-Control": "no-store", "Content-Security-Policy": REPORT_CSP,
                                 "X-Content-Type-Options": "nosniff"})


@router.post("/logout", response_model=MessageOut, summary="Sign out of the recruiter session")
def logout(request: Request, response: Response, db: Session = Depends(get_db)) -> MessageOut:
    auth_service.revoke_session(db, _token(request))
    response.delete_cookie(get_settings().recruiter_cookie_name, path="/")
    return MessageOut(message="Signed out.")
