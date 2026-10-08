"""Application errors with user-facing messages.

Every error that reaches the browser has a stable machine `code` (the UI can react to it)
and a `message` written for the candidate, not for a developer.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError, SQLAlchemyError

log = logging.getLogger(__name__)


class AppError(Exception):
    status_code = 400
    code = "BAD_REQUEST"

    def __init__(self, message: str, *, code: str | None = None, status_code: int | None = None,
                 details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code:
            self.code = code
        if status_code:
            self.status_code = status_code
        self.details = details or {}


class NotAuthenticated(AppError):
    status_code = 401
    code = "NOT_AUTHENTICATED"


class Forbidden(AppError):
    status_code = 403
    code = "FORBIDDEN"


class NotFound(AppError):
    status_code = 404
    code = "NOT_FOUND"


class Conflict(AppError):
    status_code = 409
    code = "CONFLICT"


class InvalidUpload(AppError):
    status_code = 422
    code = "INVALID_UPLOAD"


class ServiceUnavailable(AppError):
    status_code = 503
    code = "SERVICE_UNAVAILABLE"


def _body(code: str, message: str, details: dict | None = None) -> dict:
    return {"error": {"code": code, "message": message, "details": details or {}}}


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError):
        return JSONResponse(_body(exc.code, exc.message, exc.details), status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        fields = {}
        for err in exc.errors():
            loc = [str(p) for p in err.get("loc", []) if p not in ("body", "query", "path", "form")]
            fields[".".join(loc) or "request"] = err.get("msg", "Invalid value").removeprefix("Value error, ")
        return JSONResponse(
            _body("VALIDATION_ERROR", "Some of the information provided is not valid.", {"fields": fields}),
            status_code=422,
        )

    @app.exception_handler(OperationalError)
    async def _db_down(_: Request, exc: OperationalError):
        log.error("Database unavailable: %s", exc.__class__.__name__)
        return JSONResponse(
            _body("DATABASE_UNAVAILABLE", "We could not reach the database. Please try again in a moment."),
            status_code=503,
        )

    @app.exception_handler(SQLAlchemyError)
    async def _db_error(_: Request, exc: SQLAlchemyError):
        log.exception("Database error")
        return JSONResponse(
            _body("DATABASE_ERROR", "Something went wrong while saving your data. Please try again."),
            status_code=500,
        )

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        log.exception("Unhandled error")
        return JSONResponse(
            _body("INTERNAL_ERROR", "Something went wrong on our side. Please try again."),
            status_code=500,
        )
