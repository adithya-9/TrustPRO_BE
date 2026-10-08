"""Password hashing and session tokens."""
from __future__ import annotations

import hashlib
import secrets

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError


# Argon2id with the library defaults (RFC 9106 "low memory" profile: t=3, m=64 MiB, p=4),
# which satisfies the OWASP Password Storage Cheat Sheet recommendation.
_hasher = PasswordHasher()

# A real hash computed once, used to keep login timing similar for unknown emails.
_DUMMY_HASH = _hasher.hash("trustpro-timing-equaliser")


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password_hash: str | None, password: str) -> bool:
    try:
        return _hasher.verify(password_hash or _DUMMY_HASH, password) and password_hash is not None
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def needs_rehash(password_hash: str) -> bool:
    return _hasher.check_needs_rehash(password_hash)


def new_session_token() -> str:
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
