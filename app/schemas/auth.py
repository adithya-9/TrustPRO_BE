from __future__ import annotations

import re

from pydantic import field_validator

from app.schemas.common import ApiModel

EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)+$")


class Credentials(ApiModel):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def _email(cls, value: str) -> str:
        value = value.strip().lower()
        if len(value) > 255 or not EMAIL_RE.match(value):
            raise ValueError("Enter a valid email address.")
        return value


class LoginRequest(Credentials):
    @field_validator("password")
    @classmethod
    def _password(cls, value: str) -> str:
        if not value or len(value) > 128:
            raise ValueError("Enter your password.")
        return value


class RegisterRequest(Credentials):
    @field_validator("password")
    @classmethod
    def _password(cls, value: str) -> str:
        if len(value) < 8:
            raise ValueError("Use at least 8 characters.")
        if len(value) > 128:
            raise ValueError("Use at most 128 characters.")
        if not re.search(r"[A-Za-z]", value) or not re.search(r"\d", value):
            raise ValueError("Use at least one letter and one number.")
        return value


class UserOut(ApiModel):
    user_id: int
    email: str
    user_type: str
    profile_complete: bool
    id_captured: bool
    next_step: str                # PROFILE | ID_VERIFICATION | INTERVIEW
