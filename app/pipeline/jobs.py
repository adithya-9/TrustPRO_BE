"""Background job runner for report generation.

A small in-process thread pool is enough for a single-server POC: jobs are persisted in
epsoft.interview_reports (status Q/P/C/F), so a restart re-queues unfinished work, and the
API never waits for analysis. If throughput needs grow, the same run_report() function can
be called from a separate worker process without changes.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import select

from app.core.config import get_settings
from app.db.models import InterviewReport, ReportStatus
from app.db.session import session_scope
from app.pipeline.runner import run_report

log = logging.getLogger(__name__)


class ReportJobs:
    def __init__(self) -> None:
        self._pool: ThreadPoolExecutor | None = None

    def start(self) -> None:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=get_settings().analysis_job_workers,
                                            thread_name_prefix="report-job")

    def submit(self, report_id: int) -> None:
        self.start()
        self._pool.submit(run_report, report_id)
        log.info("Report queued report_id=%s", report_id)

    def recover(self) -> int:
        """Re-queue reports left queued/processing by a previous server process."""
        with session_scope() as db:
            pending = list(db.scalars(select(InterviewReport).where(
                InterviewReport.status.in_([ReportStatus.QUEUED, ReportStatus.PROCESSING]))))
            for report in pending:
                report.status = ReportStatus.QUEUED
                report.progress_pct = 0
                report.current_stage = "Queued (resumed after restart)"
            ids = [r.report_id for r in pending]
        for report_id in ids:
            self.submit(report_id)
        return len(ids)

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None


jobs = ReportJobs()
