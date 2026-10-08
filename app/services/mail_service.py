"""Outgoing email. Same providers as tl-core (mail-sender.provider = AZURE | SENDGRID), plus plain
SMTP for setups without either:

  MAIL_PROVIDER=azure     Microsoft Graph sendMail with an app registration (client credentials)
  MAIL_PROVIDER=sendgrid  SendGrid v3 API key
  MAIL_PROVIDER=smtp      any SMTP server (e.g. Office 365 / Gmail with an app password)
  MAIL_PROVIDER=none      email off (default)

Only standard-library HTTP/SMTP is used. Secrets come from .env and are never logged.
"""
from __future__ import annotations

import json
import logging
import smtplib
import urllib.parse
import urllib.request
from email.message import EmailMessage

from app.core.config import get_settings

log = logging.getLogger(__name__)
TIMEOUT_S = 30


def _post_json(url: str, payload: dict, headers: dict) -> int:
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                     headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        return response.status


def _azure_token() -> str:
    s = get_settings()
    body = urllib.parse.urlencode({"client_id": s.azure_client_id, "client_secret": s.azure_client_secret,
                                   "scope": s.azure_scope, "grant_type": "client_credentials"}).encode()
    url = s.azure_token_url.replace("{tenantId}", s.azure_tenant_id)
    request = urllib.request.Request(url, data=body, method="POST",
                                     headers={"Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        return json.loads(response.read())["access_token"]


def _send_azure(to: list[str], subject: str, html: str) -> None:
    s = get_settings()
    payload = {"message": {"subject": subject, "body": {"contentType": "HTML", "content": html},
                           "toRecipients": [{"emailAddress": {"address": a}} for a in to]},
               "saveToSentItems": False}
    url = s.azure_graph_url.replace("{fromEmail}", s.mail_from)
    _post_json(url, payload, {"Authorization": f"Bearer {_azure_token()}"})


def _send_sendgrid(to: list[str], subject: str, html: str, text: str) -> None:
    s = get_settings()
    payload = {"personalizations": [{"to": [{"email": a} for a in to]}], "from": {"email": s.mail_from},
               "subject": subject, "content": [{"type": "text/plain", "value": text},
                                               {"type": "text/html", "value": html}]}
    _post_json("https://api.sendgrid.com/v3/mail/send", payload, {"Authorization": f"Bearer {s.sendgrid_api_key}"})


def _send_smtp(to: list[str], subject: str, html: str, text: str) -> None:
    s = get_settings()
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, s.mail_from, ", ".join(to)
    msg.set_content(text)
    msg.add_alternative(html, subtype="html")
    with smtplib.SMTP(s.smtp_host, s.smtp_port, timeout=TIMEOUT_S) as smtp:
        if s.smtp_starttls:
            smtp.starttls()
        if s.smtp_username:
            smtp.login(s.smtp_username, s.smtp_password)
        smtp.send_message(msg)


def send_mail(to: list[str], subject: str, html: str, text: str) -> bool:
    """True when the provider accepted the message. Never raises: email is best-effort."""
    s = get_settings()
    provider = (s.mail_provider or "none").strip().lower()
    to = [a.strip() for a in to if a and a.strip()]
    if provider == "none" or not to:
        return False
    if not s.mail_from:
        log.error("Email not sent: MAIL_FROM is not set")
        return False
    try:
        if provider == "azure":
            _send_azure(to, subject, html)
        elif provider == "sendgrid":
            _send_sendgrid(to, subject, html, text)
        elif provider == "smtp":
            _send_smtp(to, subject, html, text)
        else:
            log.error("Email not sent: unknown MAIL_PROVIDER %r (use azure, sendgrid, smtp or none)", provider)
            return False
    except Exception as exc:  # noqa: BLE001 - report the failure, keep the caller running
        log.error("Email via %s failed: %s", provider, exc.__class__.__name__)
        return False
    log.info("Email sent via %s to %s", provider, ", ".join(to))
    return True
