"""Post-assessment report: one job, started automatically when the assessment ends.

                      frames, 1 per second
                                 |
            +--------------------+---------------------+
            v                    v                     v
      identity queue      environment queue      ID document (shown before the assessment)
            |                    |                     |
     IdentityAnalyzer    EnvironmentAnalyzer      OCR: name + DOB vs profile,
     (YuNet + ArcFace,   (YOLO11m, 8 frames per    ID photo vs profile photo
      every 3 s)          call, Streamlit rules)
            +--------------------+---------------------+
                                 v
              ID photo vs every video face -> summaries in the database
                                 v
              HTML report on disk; its link is printed in the server log,
              plus a recruiter login (link, login ID, password) for this report only

Normally the frames were already analysed while the assessment was recorded (live.py), so the
job only waits for the last few seconds and builds the report. Without a usable live analysis
(server restarted mid-assessment, live analysis off or failed) it sweeps the whole recording.

The three branches run at the same time. A failure inside one branch marks only that section
as failed; the report still completes. Only an unreadable recording fails it.
"""
from __future__ import annotations

import logging
import threading
import time

from app.ai.runtime import ModelUnavailableError
from app.db.models import IdVerification, InterviewReport, ReportStatus, utcnow
from app.db.session import session_scope
from app.pipeline import analysis
from app.pipeline import frames as frame_source
from app.pipeline.analysis import ReportFailed
from app.pipeline.live import LiveAnalysis, registry as live_registry
from app.services import html_report, recruiter_service, verification_service

log = logging.getLogger(__name__)

__all__ = ["ReportFailed", "run_report"]


class Progress:
    """Throttled progress writes to the report row."""

    def __init__(self, report_id: int) -> None:
        self.report_id = report_id
        self._last = 0.0
        self._pct = -1
        self._lock = threading.Lock()

    def update(self, pct: float, stage: str, force: bool = False, frames: int | None = None) -> None:
        pct_i = int(max(0, min(99, pct)))
        now = time.monotonic()
        with self._lock:
            if not force and (now - self._last < 1.5 or pct_i == self._pct):
                return
            self._last, self._pct = now, pct_i
        with session_scope() as db:
            report = db.get(InterviewReport, self.report_id)
            if report:
                report.progress_pct = pct_i
                report.current_stage = stage
                if frames is not None:
                    report.frames_analyzed = frames


def run_report(report_id: int) -> None:
    started = time.monotonic()
    with session_scope() as db:
        report = db.get(InterviewReport, report_id)
        if report is None or report.status == ReportStatus.COMPLETED:
            return
        report.status = ReportStatus.PROCESSING
        report.started_at = utcnow()
        report.progress_pct = 0
        report.current_stage = "Preparing analysis"
        report.error_message = None
        interval_ms = report.sample_interval_ms
        interview_id = report.interview_id
    live = live_registry.take(interview_id)
    log.info("Report generation started report_id=%s (%s)", report_id,
             "analysed while recording" if live else "full analysis of the recording")
    try:
        path = _run(report_id, interview_id, interval_ms, Progress(report_id), started, live)
        log.info("Report generation completed report_id=%s in %.1fs", report_id, time.monotonic() - started)
        log.info("REPORT READY (report %s): %s", report_id, path.resolve().as_uri())
        _issue_recruiter_access(report_id)
    except ReportFailed as exc:
        log.warning("Report generation failed report_id=%s: %s", report_id, exc.user_message)
        _mark_failed(report_id, exc.user_message)
    except ModelUnavailableError as exc:
        log.error("Report generation failed report_id=%s: model unavailable: %s", report_id, exc)
        _mark_failed(report_id, "An analysis model is not available on the server.")
    except Exception:
        log.exception("Report generation crashed report_id=%s", report_id)
        _mark_failed(report_id, "Something went wrong while analysing the assessment.")


def _issue_recruiter_access(report_id: int) -> None:
    """A recruiter login for this report only, printed in the terminal. Never fails the report."""
    try:
        with session_scope() as db:
            access = recruiter_service.issue_access(db, report_id)
        recruiter_service.print_access(access)
    except Exception:
        log.exception("Recruiter access could not be created report_id=%s", report_id)


def _mark_failed(report_id: int, message: str) -> None:
    try:
        with session_scope() as db:
            report = db.get(InterviewReport, report_id)
            if report:
                report.status = ReportStatus.FAILED
                report.error_message = message[:500]
                report.current_stage = "Failed"
                report.completed_at = utcnow()
    except Exception:
        log.exception("Could not record report failure report_id=%s", report_id)


def _use_live(live: LiveAnalysis, expected: int, progress: Progress) -> bool:
    """Wait for the live analysis to cover the end of the recording; True if it can be used."""
    live.finish()
    progress.update(2, "Finishing the analysis", force=True)
    while not live.done.wait(1.0):
        progress.update(2 + 90 * min(1.0, live.frames / expected), "Finishing the analysis", frames=live.frames)
    if live.ok:
        return True
    log.warning("Live analysis not usable interview_id=%s (%s); analysing the whole recording",
                live.interview_id, live.error or "no frames")
    return False


def _run(report_id: int, interview_id: int, interval_ms: int, progress: Progress, started: float,
         live: LiveAnalysis | None):
    timings: dict = {}
    t0 = time.monotonic()
    inputs = analysis.load_inputs(interview_id)
    video_path, duration_ms = inputs["video_path"], inputs["duration_ms"]
    if not video_path.exists() or video_path.stat().st_size == 0:
        raise ReportFailed("The assessment recording file is missing or empty.")
    try:
        video_info = frame_source.probe(video_path)
    except frame_source.FrameExtractionError as exc:
        raise ReportFailed(str(exc)) from exc
    expected = max(1, duration_ms // interval_ms + 1)

    if live is not None and _use_live(live, expected, progress):
        identity, environment, sections, id_result = live.identity, live.environment, live.sections, live.id_result
        timings["mode"] = "live"
        timings["wait_after_end_s"] = round(time.monotonic() - t0, 2)
    else:
        if live is not None:
            live.cancel()
        timings["mode"] = "full"
        identity, environment, sections = analysis.make_analyzers(inputs, interval_ms)
        id_result: dict = {}
        id_thread = threading.Thread(target=analysis.analyse_id_document, args=(inputs, id_result),
                                     name="id-document", daemon=True)
        id_thread.start()
        progress.update(2, "Analysing identity and environment", force=True)
        error = analysis.analyse_frames(
            frame_source.sample_frames(video_path, interval_ms), identity, environment, analysis.face_every(interval_ms),
            lambda n: progress.update(2 + 90 * min(1.0, n / expected), "Analysing identity and environment", frames=n))
        id_thread.join()
        if error is not None:
            raise ReportFailed(f"The recording could not be read: {error}")
    timings["analysis_s"] = round(time.monotonic() - t0, 2)

    # ---------------------------------------------------------------- summaries
    progress.update(94, "Building the report", force=True)
    reading = id_result.get("reading")
    id_evidence = []
    if identity is not None:
        try:
            sections["identity"], id_evidence = identity.finalize(reading.portrait_embedding if reading else None)
        except Exception:
            log.exception("Identity summary failed report_id=%s", report_id)
            sections["identity"] = {"status": "failed", "error": "The face comparison could not be summarised."}
    if environment is not None:
        sections["environment"] = environment.summary()

    with session_scope() as db:
        report = db.get(InterviewReport, report_id)
        if reading is not None:
            record = db.get(IdVerification, inputs["verification_id"])
            id_document = {"status": "completed",
                           **verification_service.save_analysis(db, record, inputs["owner_user_id"], reading)}
        else:
            id_document = id_result.get("summary", {"status": "failed", "error": "Not analysed."})
        identity_section = dict(sections.get("identity") or {"status": "failed", "error": "Not analysed."})
        identity_section.pop("timeline", None)
        identity_section["id_document"] = id_document
        timings["total_s"] = round(time.monotonic() - started, 2)
        report.identity_summary = identity_section
        report.environment_summary = sections.get("environment")
        report.frames_analyzed = environment.frames_analyzed if environment else 0
        report.processing_info = {"sample_interval_ms": interval_ms, "video": video_info, "timings": timings,
                                  "models": {"face": "YuNet + ArcFace", "ocr": "RapidOCR (PP-OCRv5)",
                                             "environment": environment.model_name if environment else None}}
        report.status = ReportStatus.COMPLETED
        report.progress_pct = 100
        report.current_stage = "Completed"
        report.completed_at = utcnow()

    path = inputs["folder"] / f"report_{report_id}.html"
    html_report.write(path, inputs=inputs, identity=identity_section,
                      environment=sections.get("environment") or {"status": "failed", "error": "Not analysed."},
                      environment_events=environment.events if environment else [],
                      identity_evidence=id_evidence, report_id=report_id)
    return path
