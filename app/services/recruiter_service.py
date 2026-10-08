"""Recruiter access to a single candidate's report.

When a report completes, issue_access() creates a recruiter login - a random login ID and a
random password - that can open only that report, for recruiter_access_days. The plaintext
password exists only in the terminal print-out; the database keeps its Argon2id hash. Issuing
access again for the same report replaces the previous login.
"""
from __future__ import annotations

import logging
import secrets
import socket
from html import escape
from urllib.parse import urlencode
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.core import security
from app.core.config import get_settings
from app.core.errors import AppError, Conflict, NotAuthenticated, NotFound
from app.db.models import (InterviewReport, ReportAccess, ReportStatus, User, UserSession, UserType,
                           utcnow)
from app.services import auth_service, mail_service
from app.services.storage import get_storage

log = logging.getLogger(__name__)

LOGIN_DOMAIN = "trustpro.local"
# No look-alike characters (0/O, 1/l/I), so the password can be read out or retyped safely.
_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnpqrstuvwxyz23456789"


@dataclass(frozen=True)
class IssuedAccess:
    report_id: int
    login_id: str
    password: str
    expires_at: datetime
    login_url: str
    candidate_name: str = "Candidate"

    @property
    def autofill_url(self) -> str:
        """Login link that fills in the login ID and password. They travel in the URL fragment
        (after #), which browsers never send to a server, so they stay out of server logs."""
        return f"{self.login_url}#" + urlencode({"login": self.login_id, "password": self.password})


def _random_login_id() -> str:
    return f"rec-{''.join(secrets.choice('abcdefghjkmnpqrstuvwxyz23456789') for _ in range(8))}@{LOGIN_DOMAIN}"


def _random_password() -> str:
    return "-".join("".join(secrets.choice(_ALPHABET) for _ in range(4)) for _ in range(3))


def _lan_address() -> str:
    """This machine's address on the local network (no packet is sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return "localhost"


def app_base_url() -> str:
    s = get_settings()
    return (s.public_app_url or f"http://{_lan_address()}:{s.ui_port}").rstrip("/")


def issue_access(db: Session, report_id: int) -> IssuedAccess:
    report = db.get(InterviewReport, report_id)
    if report is None:
        raise NotFound("Report not found.")
    db.execute(delete(ReportAccess).where(ReportAccess.report_id == report_id))   # trigger drops the old login
    password = _random_password()
    login_id = _random_login_id()
    while db.scalar(select(User.user_id).where(User.email == login_id)):
        login_id = _random_login_id()
    user = User(email=login_id, password_hash=security.hash_password(password), user_type=UserType.REVIEWER)
    db.add(user)
    db.flush()
    expires_at = utcnow() + timedelta(days=get_settings().recruiter_access_days)
    db.add(ReportAccess(report_id=report_id, user_id=user.user_id, expires_at=expires_at))
    db.commit()
    log.info("Recruiter access issued report_id=%s user_id=%s", report_id, user.user_id)
    return IssuedAccess(report_id=report_id, login_id=login_id, password=password, expires_at=expires_at,
                        login_url=f"{app_base_url()}/recruiter/login",
                        candidate_name=report.interview.candidate.full_name or "Candidate")


def email_access(access: IssuedAccess) -> bool:
    """Email the recruiter login to RECRUITER_EMAIL. False when email is off or failed."""
    to = [a for a in get_settings().recruiter_email.split(",") if a.strip()]
    if not to:
        return False
    name = escape(access.candidate_name)
    until = f"{access.expires_at:%d %b %Y %H:%M} UTC"
    link = escape(access.autofill_url, quote=True)
    html = f"""<div style="font-family:Segoe UI,Arial,sans-serif;color:#172033;max-width:560px">
  <h2 style="margin:0 0 8px">Assessment report ready: {name}</h2>
  <p style="color:#4b5563">The TrustPRO report for <b>{name}</b> (report {access.report_id}) is ready.
     The button opens the report login with your details already filled in.</p>
  <p style="margin:24px 0"><a href="{link}" style="background:#2f5fe0;color:#fff;padding:12px 22px;
     border-radius:10px;text-decoration:none;font-weight:600">Generate Report</a></p>
  <table style="font-size:14px;border-collapse:collapse">
    <tr><td style="padding:4px 12px 4px 0;color:#6b7280">Login ID</td><td><b>{escape(access.login_id)}</b></td></tr>
    <tr><td style="padding:4px 12px 4px 0;color:#6b7280">Password</td><td><b>{escape(access.password)}</b></td></tr>
    <tr><td style="padding:4px 12px 4px 0;color:#6b7280">Valid until</td><td>{until}</td></tr>
  </table>
  <p style="color:#6b7280;font-size:12px;margin-top:24px">This login opens only this candidate's report.
     Do not forward this email.</p>
</div>"""
    text = (f"The TrustPRO report for {access.candidate_name} (report {access.report_id}) is ready.\n\n"
            f"Open: {access.autofill_url}\nLogin ID: {access.login_id}\nPassword: {access.password}\n"
            f"Valid until {until}\n")
    return mail_service.send_mail(to, f"TrustPRO report ready: {access.candidate_name}", html, text)


def print_access(access: IssuedAccess) -> None:
    """Shown in the server terminal only (printed, not logged, so the password stays out of log files)."""
    line = "=" * 64
    print(f"\n{line}\n RECRUITER ACCESS - report {access.report_id}"
          f"\n   Link     : {access.login_url}"
          f"\n   Login ID : {access.login_id}"
          f"\n   Password : {access.password}"
          f"\n   Valid until {access.expires_at:%d %b %Y %H:%M} UTC\n{line}\n", flush=True)


def _active_access(db: Session, user: User) -> ReportAccess:
    if user.user_type != UserType.REVIEWER:
        raise NotAuthenticated("The login ID or password is incorrect.", code="INVALID_CREDENTIALS")
    access = db.scalar(select(ReportAccess).where(ReportAccess.user_id == user.user_id))
    if access is None:
        raise NotAuthenticated("The login ID or password is incorrect.", code="INVALID_CREDENTIALS")
    if access.expires_at <= utcnow():
        raise AppError("This report link has expired.", code="ACCESS_EXPIRED", status_code=401)
    return access


def login(db: Session, login_id: str, password: str) -> tuple[str, ReportAccess]:
    """Check the credentials; returns a new session token limited to the access expiry."""
    user = auth_service.authenticate(db, login_id, password)
    access = _active_access(db, user)
    token = security.new_session_token()
    db.add(UserSession(user_id=user.user_id, token_hash=security.hash_token(token),
                       expires_at=min(utcnow() + timedelta(hours=get_settings().session_ttl_hours), access.expires_at)))
    access.last_login_at = utcnow()
    db.commit()
    log.info("Recruiter signed in report_id=%s user_id=%s", access.report_id, user.user_id)
    return token, access


def resolve(db: Session, token: str | None) -> ReportAccess:
    return _active_access(db, auth_service.resolve_session(db, token))


def report_html_path(report: InterviewReport) -> Path:
    recording = report.interview.recording
    if recording is None:
        raise NotFound("The report file was not found.")
    return get_storage().path_for(recording.storage_key).parent / f"report_{report.report_id}.html"


def report_html(db: Session, access: ReportAccess) -> str:
    report = db.get(InterviewReport, access.report_id)
    if report is None:
        raise NotFound("Report not found.")
    if report.status != ReportStatus.COMPLETED:
        raise Conflict("The report is not ready yet.", code="REPORT_NOT_READY")
    path = report_html_path(report)
    if not path.exists():
        raise NotFound("The report file was not found.")
    return path.read_text(encoding="utf-8")


def candidate_name(access: ReportAccess) -> str:
    return access.report.interview.candidate.full_name or "Candidate"
