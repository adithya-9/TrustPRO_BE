"""Create (or replace) the recruiter login for completed reports and print it.

New reports get a login automatically when they complete; use this for older reports or when
the printed password was lost. A new login replaces the previous one for that report.

    python -m scripts.recruiter_access 45          # one report
    python -m scripts.recruiter_access 45 46       # several
    python -m scripts.recruiter_access --latest    # the most recent completed report
"""
from __future__ import annotations

import argparse
import sys

from sqlalchemy import select

from app.db.models import InterviewReport, ReportStatus
from app.db.session import session_scope
from app.services import recruiter_service


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("report_ids", nargs="*", type=int)
    parser.add_argument("--latest", action="store_true", help="the most recent completed report")
    args = parser.parse_args()

    with session_scope() as db:
        ids = list(args.report_ids)
        if args.latest:
            latest = db.scalar(select(InterviewReport.report_id).where(InterviewReport.status == ReportStatus.COMPLETED)
                               .order_by(InterviewReport.completed_at.desc()).limit(1))
            if latest is None:
                print("No completed report found.")
                return 1
            ids.append(latest)
        if not ids:
            parser.print_help()
            return 1
        for report_id in ids:
            report = db.get(InterviewReport, report_id)
            if report is None or report.status != ReportStatus.COMPLETED:
                print(f"Report {report_id}: not found or not completed - skipped.")
                continue
            access = recruiter_service.issue_access(db, report_id)
            recruiter_service.print_access(access)
            if recruiter_service.email_access(access):
                print(f"Report {report_id}: login emailed to RECRUITER_EMAIL.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
