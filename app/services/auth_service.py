"""Email/password authentication with opaque server-side sessions (no JWT).

The browser holds a random token in an HttpOnly cookie; the database stores only its
SHA-256. Logging out or expiring a session is a single row update.
"""
from __future__ import annotations

import logging
from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core import security
from app.core.config import get_settings
from app.core.errors import AppError, Conflict, NotAuthenticated
from app.db.models import Candidate, User, UserSession, UserType, utcnow

log = logging.getLogger(__name__)
LOCKOUT_WINDOW = timedelta(minutes=15)
LAST_ACTIVE_RESOLUTION = timedelta(minutes=5)


def register(db: Session, email: str, password: str) -> User:
    if db.scalar(select(User.user_id).where(User.email == email)):
        raise Conflict("An account with this email already exists. Please sign in instead.", code="EMAIL_TAKEN")
    user = User(email=email, password_hash=security.hash_password(password), user_type=UserType.CANDIDATE)
    db.add(user)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        raise Conflict("An account with this email already exists. Please sign in instead.", code="EMAIL_TAKEN")
    db.add(Candidate(user_id=user.user_id))
    db.commit()
    log.info("User registered user_id=%s", user.user_id)
    return user


def authenticate(db: Session, email: str, password: str) -> User:
    settings = get_settings()
    user = db.scalar(select(User).where(User.email == email))
    if user and user.failed_attempts >= settings.max_failed_logins and utcnow() - user.updated_at < LOCKOUT_WINDOW:
        raise AppError("Too many unsuccessful sign-in attempts. Please wait 15 minutes and try again.",
                       code="ACCOUNT_LOCKED", status_code=429)
    if not security.verify_password(user.password_hash if user else None, password):
        if user:
            user.failed_attempts += 1
            db.commit()
            log.info("Failed login user_id=%s attempts=%s", user.user_id, user.failed_attempts)
        raise NotAuthenticated("The email or password is incorrect.", code="INVALID_CREDENTIALS")
    if user.is_disabled:
        raise AppError("This account has been disabled. Please contact support.", code="ACCOUNT_DISABLED", status_code=403)
    user.failed_attempts = 0
    user.last_active = utcnow()
    if security.needs_rehash(user.password_hash):
        user.password_hash = security.hash_password(password)
    db.commit()
    return user


def create_session(db: Session, user: User) -> str:
    token = security.new_session_token()
    db.add(UserSession(
        user_id=user.user_id,
        token_hash=security.hash_token(token),
        expires_at=utcnow() + timedelta(hours=get_settings().session_ttl_hours),
    ))
    db.commit()
    log.info("Session created user_id=%s", user.user_id)
    return token


def resolve_session(db: Session, token: str | None) -> User:
    if not token:
        raise NotAuthenticated("Please sign in to continue.")
    session = db.scalar(select(UserSession).where(UserSession.token_hash == security.hash_token(token)))
    now = utcnow()
    if session is None or session.revoked_at is not None or session.expires_at <= now:
        raise NotAuthenticated("Your session has expired. Please sign in again.", code="SESSION_EXPIRED")
    user = db.get(User, session.user_id)
    if user is None or user.is_disabled:
        raise NotAuthenticated("Please sign in to continue.")
    if user.last_active is None or now - user.last_active > LAST_ACTIVE_RESOLUTION:
        user.last_active = now
        db.commit()
    return user


def revoke_session(db: Session, token: str | None) -> None:
    if not token:
        return
    session = db.scalar(select(UserSession).where(UserSession.token_hash == security.hash_token(token)))
    if session and session.revoked_at is None:
        session.revoked_at = utcnow()
        db.commit()
