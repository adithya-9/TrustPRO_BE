"""Send a test email with the MAIL_* settings from .env, to check the configuration.

    python -m scripts.send_test_email                 # to RECRUITER_EMAIL
    python -m scripts.send_test_email you@company.com
"""
from __future__ import annotations

import logging
import sys

from app.core.config import get_settings
from app.services import mail_service


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    s = get_settings()
    to = sys.argv[1:] or [a for a in s.recruiter_email.split(",") if a.strip()]
    print(f"MAIL_PROVIDER={s.mail_provider!r}  MAIL_FROM={s.mail_from!r}  to={to}")
    if (s.mail_provider or "none").lower() == "none":
        print("Email is off: set MAIL_PROVIDER (azure | sendgrid | smtp) and its settings in .env.")
        return 1
    if not to:
        print("No recipient: set RECRUITER_EMAIL in .env or pass an address.")
        return 1
    ok = mail_service.send_mail(to, "TrustPRO test email", "<p>TrustPRO email is working.</p>",
                                "TrustPRO email is working.")
    print("SENT - check the inbox (and spam)." if ok else "FAILED - see the error line above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
