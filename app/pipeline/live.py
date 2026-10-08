"""Analysis while the assessment is being recorded, so the report is ready right after it ends.

The browser uploads the recording in 3-second WebM chunks that the server appends to one file.
When the assessment starts, a LiveAnalysis follows that growing file: it decodes the WebM
stream as the bytes arrive (PyAV/FFmpeg, through GrowingFile, whose read() waits for more data)
and runs exactly the same analysis as the full sweep - 1 frame per second, YOLO11m with the
Streamlit rules, faces every 3 s, the government-ID check once. Nothing is shown to the
candidate; the browser only records and uploads, as before.

When the assessment ends, the report job waits for the last few seconds to be analysed and
then builds the report from these results. If the live analysis is missing (server restarted
mid-assessment) or failed, the report falls back to the full sweep of the recording.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from app.core.config import get_settings
from app.pipeline import analysis

log = logging.getLogger(__name__)

POLL_S = 0.25
MAX_WIDTH = 960     # same frame size as the full sweep (frames.sample_frames)


class LiveStopped(RuntimeError):
    """The live analysis was cancelled, or the upload went quiet for too long."""


class GrowingFile:
    """Read-only file object over a recording that is still being uploaded. read() waits for
    more bytes until the recording is marked finished (then returns b"" at the end)."""

    def __init__(self, path: Path, finished: threading.Event, stop: threading.Event, idle_timeout_s: float) -> None:
        self.path = path
        self.finished = finished
        self.stop = stop
        self.idle_timeout_s = idle_timeout_s
        self._file = None

    def read(self, size: int = -1) -> bytes:
        size = size if size and size > 0 else 1 << 20
        waited = 0.0
        while True:
            if self._file is None and self.path.exists():
                self._file = open(self.path, "rb")
            if self._file is not None:
                data = self._file.read(size)
                if data:
                    return data
            if self.stop.is_set():
                raise LiveStopped("cancelled")
            if self.finished.is_set():
                # The last chunk was appended before "finished" was set; read whatever is left.
                return self._file.read(size) if self._file is not None else b""
            if waited >= self.idle_timeout_s:
                raise LiveStopped(f"no new recording data for {int(self.idle_timeout_s)} s")
            time.sleep(POLL_S)
            waited += POLL_S

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def sample_growing_frames(source: GrowingFile, interval_ms: int, info: dict) -> Iterator[tuple[int, np.ndarray]]:
    """(offset_ms, BGR frame) at interval_ms, decoded while the file grows. Same sampling rule as
    frames.sample_frames: the first frame at or after each interval boundary."""
    import av

    container = av.open(source, mode="r", options={"probesize": "262144", "analyzeduration": "1000000"})
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        ctx = stream.codec_context
        info.update(width=ctx.width, height=ctx.height, codec=ctx.name)
        next_ts = 0
        last_ts = -1
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            ts_ms = max(last_ts, int(frame.time * 1000))      # never go backwards
            last_ts = ts_ms
            if ts_ms + 1 < next_ts:
                continue
            image = frame.to_ndarray(format="bgr24")
            h, w = image.shape[:2]
            if w > MAX_WIDTH:
                image = cv2.resize(image, (MAX_WIDTH, int(h * MAX_WIDTH / w)), interpolation=cv2.INTER_AREA)
            yield ts_ms, image
            next_ts = (ts_ms // interval_ms + 1) * interval_ms
    finally:
        container.close()


class LiveAnalysis:
    def __init__(self, interview_id: int, interval_ms: int) -> None:
        self.interview_id = interview_id
        self.interval_ms = interval_ms
        self.finished = threading.Event()   # the assessment ended; the file will not grow any more
        self.stop = threading.Event()       # cancel
        self.done = threading.Event()       # analysis thread has exited
        self.error: str | None = None
        self.frames = 0
        self.finished_at: float | None = None
        self.done_at: float | None = None
        self.inputs: dict | None = None
        self.identity = self.environment = None
        self.sections: dict = {}
        self.id_result: dict = {}
        self.video_info: dict = {}
        self._thread = threading.Thread(target=self._run, name=f"live-{interview_id}", daemon=True)

    @property
    def ok(self) -> bool:
        return self.done.is_set() and self.error is None and self.frames > 0

    def start(self) -> None:
        self._thread.start()

    def finish(self) -> None:
        if not self.finished.is_set():
            self.finished_at = time.monotonic()
            self.finished.set()

    def cancel(self) -> None:
        self.stop.set()

    def _progress(self, n: int) -> None:
        self.frames = max(self.frames, n)

    def _run(self) -> None:
        source = None
        stopped = False
        try:
            self.inputs = analysis.load_inputs(self.interview_id)
            self.identity, self.environment, self.sections = analysis.make_analyzers(self.inputs, self.interval_ms)
            id_thread = threading.Thread(target=analysis.analyse_id_document, args=(self.inputs, self.id_result),
                                         name=f"live-id-{self.interview_id}", daemon=True)
            id_thread.start()
            source = GrowingFile(self.inputs["video_path"], self.finished, self.stop,
                                 get_settings().live_analysis_idle_timeout_s)
            error = analysis.analyse_frames(sample_growing_frames(source, self.interval_ms, self.video_info),
                                            self.identity, self.environment, analysis.face_every(self.interval_ms),
                                            self._progress)
            id_thread.join()
            if error is not None:
                raise error
            if self.environment is not None:
                self.frames = self.environment.frames_analyzed
            if self.frames == 0:
                self.error = "no frames could be read while recording"
            log.info("Live analysis finished interview_id=%s frames=%s", self.interview_id, self.frames)
        except LiveStopped as exc:
            self.error = str(exc)
            stopped = True
            log.info("Live analysis stopped interview_id=%s: %s", self.interview_id, exc)
        except Exception as exc:  # noqa: BLE001 - the report falls back to the full sweep
            self.error = f"{exc.__class__.__name__}: {exc}"
            log.warning("Live analysis failed interview_id=%s: %s", self.interview_id, self.error)
        finally:
            if source is not None:
                source.close()
            self.done_at = time.monotonic()
            self.done.set()
            if stopped and not self.finished.is_set():
                # Upload went quiet (or cancelled): forget it, so a later chunk starts a fresh one.
                # A failed analysis stays registered, so the report knows to fall back.
                registry.discard(self.interview_id, self)


class LiveRegistry:
    """Live analyses of the assessments currently being recorded in this server process."""

    def __init__(self) -> None:
        self._items: dict[int, LiveAnalysis] = {}
        self._lock = threading.Lock()

    def ensure(self, interview_id: int, interval_ms: int) -> None:
        """Start following this recording if nothing is following it yet (idempotent)."""
        if not get_settings().live_analysis_enabled:
            return
        with self._lock:
            if interview_id in self._items:
                return
            live = LiveAnalysis(interview_id, interval_ms)
            self._items[interview_id] = live
        live.start()
        log.info("Live analysis started interview_id=%s", interview_id)

    def finish(self, interview_id: int) -> None:
        with self._lock:
            live = self._items.get(interview_id)
        if live is not None:
            live.finish()

    def cancel(self, interview_id: int) -> None:
        with self._lock:
            live = self._items.pop(interview_id, None)
        if live is not None:
            live.cancel()

    def take(self, interview_id: int) -> LiveAnalysis | None:
        """Hand the live analysis to the report job (removes it from the registry)."""
        with self._lock:
            return self._items.pop(interview_id, None)

    def discard(self, interview_id: int, live: LiveAnalysis) -> None:
        with self._lock:
            if self._items.get(interview_id) is live:
                del self._items[interview_id]

    def shutdown(self) -> None:
        with self._lock:
            items, self._items = list(self._items.values()), {}
        for live in items:
            live.cancel()


registry = LiveRegistry()
