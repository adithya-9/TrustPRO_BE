"""Identity consistency across the interview recording.

For every sampled frame the largest face (the person closest to the camera) is embedded
and compared with the stored references (profile photo, uploaded ID photo). The result is a
set of similarity measurements, sustained low-similarity segments, and evidence frames -
not a verdict. The report states plainly that a person makes the final interpretation.
"""
from __future__ import annotations

import logging
import statistics
import threading
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.ai.face import FaceEngine
from app.core.config import get_settings
from app.services.storage import encode_jpeg

log = logging.getLogger(__name__)

MIN_FACE_FRAMES = 5
SEGMENT_MIN_FRAMES = 3
MAX_SEGMENT_EVIDENCE = 5
REPRESENTATIVE_FRAMES = 0   # only meaningful frames (lowest, highest, low-similarity periods)
MAX_KEPT_THUMBS = 400
THUMB_WIDTH = 640


@dataclass
class FrameIdentity:
    offset_ms: int
    faces: int
    sim_profile: float | None = None
    sim_id: float | None = None
    box: tuple[int, int, int, int] | None = None
    embedding: np.ndarray | None = None   # float16; compared with the ID photo once it is analysed


@dataclass
class EvidenceCandidate:
    kind: str                 # IDENTITY | ENVIRONMENT
    label: str
    offset_ms: int
    jpeg: bytes
    confidence: float | None
    model_result: dict = field(default_factory=dict)


def consistency_label(similarity: float | None, threshold: float, uncertain_floor: float | None = None) -> str:
    if similarity is None:
        return "UNAVAILABLE"
    if similarity >= threshold:
        return "CONSISTENT"
    if uncertain_floor is not None and similarity >= uncertain_floor:
        return "INCONCLUSIVE"
    return "NOT_CONSISTENT"


def id_photo_label(similarity: float | None) -> str:
    s = get_settings()
    return consistency_label(similarity, s.id_photo_match_threshold, s.id_photo_uncertain_floor)


def _small(frame: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    if w <= THUMB_WIDTH:
        return frame.copy(), 1.0
    scale = THUMB_WIDTH / w
    return cv2.resize(frame, (THUMB_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA), scale


def _rescaled(item: tuple[FrameIdentity, tuple[np.ndarray, float]]) -> tuple[np.ndarray, FrameIdentity]:
    """The stored small frame, and a copy of the record with its face box scaled to match."""
    record, (small, scale) = item
    box = None if record.box is None else tuple(int(v * scale) for v in record.box)
    return small, FrameIdentity(record.offset_ms, record.faces, record.sim_profile, record.sim_id, box)


class IdentityAnalyzer:
    def __init__(self, engine: FaceEngine, profile_embedding: np.ndarray | None, interval_ms: int) -> None:
        if profile_embedding is None:
            raise ValueError("No face could be read from the profile photo, so the video cannot be compared with it.")
        self.engine = engine
        self.profile = profile_embedding
        self.threshold = engine.match_threshold
        self.interval_ms = interval_ms
        self.records: list[FrameIdentity] = []
        self.failed_frames = 0
        self._thumbs: dict[int, bytes] = {}
        # Best/worst frames are kept as small raw copies and JPEG-encoded once at the end;
        # encoding on every improvement was a measurable cost early in long interviews.
        self._best: tuple[FrameIdentity, tuple[np.ndarray, float]] | None = None
        self._worst: tuple[FrameIdentity, tuple[np.ndarray, float]] | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ per frame
    def process(self, offset_ms: int, frame: np.ndarray) -> None:
        try:
            faces = self.engine.detect(frame)
            record = FrameIdentity(offset_ms, len(faces))
            if faces:
                main = faces[0]
                embedding = self.engine.embed(frame, main)
                record.sim_profile = self.engine.similarity(embedding, self.profile)
                record.embedding = embedding.astype(np.float16)
                record.box = main.box
        except Exception:
            log.exception("Identity analysis failed on frame at %s ms", offset_ms)
            with self._lock:
                self.failed_frames += 1
            return
        keep = record.sim_profile is not None and (record.sim_profile < self.threshold or record.faces > 1)
        thumb = None
        if record.box is not None and (keep or offset_ms % (self.interval_ms * 20) < self.interval_ms
                                       or len(self._thumbs) < 20):
            thumb = self._annotate(frame, record)
        with self._lock:
            self.records.append(record)
            if thumb is not None and len(self._thumbs) < MAX_KEPT_THUMBS:
                self._thumbs[offset_ms] = thumb
            self._track_extremes(record, frame)

    def _track_extremes(self, record: FrameIdentity, frame: np.ndarray) -> None:
        if record.sim_profile is None:
            return
        if self._best is None or record.sim_profile > self._best[0].sim_profile:
            self._best = (record, _small(frame))
        if self._worst is None or record.sim_profile < self._worst[0].sim_profile:
            self._worst = (record, _small(frame))

    def _annotate(self, frame: np.ndarray, record: FrameIdentity) -> bytes:
        out = frame.copy()
        if record.box is not None:
            x, y, w, h = record.box
            ok = record.sim_profile is not None and record.sim_profile >= self.threshold
            colour = (94, 197, 34) if ok else (11, 158, 245)
            thickness = max(2, frame.shape[1] // 320)
            cv2.rectangle(out, (x, y), (x + w, y + h), colour, thickness)
            if record.sim_profile is not None:
                label = f"similarity {record.sim_profile:.2f}"
                scale = max(0.5, frame.shape[1] / 1400)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
                cv2.rectangle(out, (x, max(0, y - th - 8)), (x + tw + 8, y), colour, -1)
                cv2.putText(out, label, (x + 4, y - 5), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
        return encode_jpeg(out, quality=82, max_side=THUMB_WIDTH)

    # ------------------------------------------------------------------ summary
    def _segments(self, records: list[FrameIdentity]) -> list[dict]:
        """Sustained stretches where the visible face was below the similarity threshold.
        One frame without a face inside a stretch does not break it."""
        segments, current, gap = [], [], 0
        for r in records:
            if r.sim_profile is None:
                gap += 1
                if gap > 1 and current:
                    segments.append(current)
                    current, gap = [], 0
                continue
            if r.sim_profile < self.threshold:
                current.append(r)
                gap = 0
            else:
                if current:
                    segments.append(current)
                current, gap = [], 0
        if current:
            segments.append(current)
        out = []
        for seg in segments:
            if len(seg) < SEGMENT_MIN_FRAMES:
                continue
            lowest = min(seg, key=lambda r: r.sim_profile)
            out.append({
                "start_ms": seg[0].offset_ms, "end_ms": seg[-1].offset_ms + self.interval_ms,
                "frames": len(seg), "min_similarity": round(lowest.sim_profile, 4),
                "median_similarity": round(statistics.median(r.sim_profile for r in seg), 4),
                "_lowest_offset": lowest.offset_ms,
            })
        return out

    def finalize(self, id_capture_embedding: np.ndarray | None) -> tuple[dict, list[EvidenceCandidate]]:
        """id_capture_embedding: the portrait on the ID shown before the interview (analysed in
        parallel with this sweep), compared here with every face frame."""
        records = sorted(self.records, key=lambda r: r.offset_ms)
        if id_capture_embedding is not None:
            for r in records:
                if r.embedding is not None:
                    r.sim_id = self.engine.similarity(r.embedding.astype(np.float32), id_capture_embedding)
        face_frames = [r for r in records if r.sim_profile is not None]
        t = self.threshold

        settings = get_settings()

        def video_stats(attr: str, threshold: float, uncertain_floor: float | None = None) -> dict:
            values = [getattr(r, attr) for r in face_frames if getattr(r, attr) is not None]
            if not values:
                return {"similarity": None, "frames_compared": 0, "consistent_frames": 0,
                        "consistent_pct": None, "result": "UNAVAILABLE", "threshold": threshold}
            consistent = sum(1 for v in values if v >= threshold)
            pct = round(100.0 * consistent / len(values), 1)
            median = statistics.median(values)
            if len(values) < MIN_FACE_FRAMES:
                result = "INCONCLUSIVE"
            elif pct >= 80:
                result = "CONSISTENT"
            elif uncertain_floor is not None and median >= uncertain_floor:
                result = "INCONCLUSIVE"       # ID photo vs video: low but not clearly different
            elif pct < 40:
                result = "NOT_CONSISTENT"
            else:
                result = "MIXED"
            return {"similarity": round(median, 4), "frames_compared": len(values),
                    "consistent_frames": consistent, "consistent_pct": pct, "result": result, "threshold": threshold,
                    "min_similarity": round(min(values), 4), "max_similarity": round(max(values), 4)}

        profile_id = (round(self.engine.similarity(self.profile, id_capture_embedding), 4)
                      if id_capture_embedding is not None else None)
        profile_video = video_stats("sim_profile", t)
        id_video = video_stats("sim_id", settings.id_photo_match_threshold, settings.id_photo_uncertain_floor)
        comparisons = [
            {"pair": "PROFILE_ID", "label": "Profile photo and government ID photo",
             "similarity": profile_id, "result": id_photo_label(profile_id), "threshold": settings.id_photo_match_threshold},
            {"pair": "PROFILE_VIDEO", "label": "Profile photo and interview video", **profile_video},
            {"pair": "ID_VIDEO", "label": "Government ID photo and interview video", **id_video},
        ]
        segments = self._segments(records)

        overall = profile_video["result"]
        n = profile_video["frames_compared"]
        if overall in ("UNAVAILABLE", "INCONCLUSIVE"):
            explanation = (f"A face was visible in only {n} analysed frame{'s' if n != 1 else ''}, which is not "
                           "enough for a reliable comparison.")
        else:
            explanation = (f"In {profile_video['consistent_pct']:.0f}% of the {n} frames where a face was visible, "
                           f"the face was consistent with the profile photo (median similarity "
                           f"{profile_video['similarity']:.2f}; consistency threshold {t:.2f}).")
            if segments:
                explanation += (f" {len(segments)} sustained period{'s' if len(segments) != 1 else ''} of low "
                                "similarity were found and are shown as evidence for review.")

        # Similarity over time for the report chart (downsampled to <= 300 points).
        step = max(1, len(records) // 300)
        timeline = [[r.offset_ms, None if r.sim_profile is None else round(r.sim_profile, 3), r.faces]
                    for r in records[::step]]

        evidence: list[EvidenceCandidate] = []

        def model_result(r: FrameIdentity, reason: str) -> dict:
            return {"reason": reason, "similarity_profile": None if r.sim_profile is None else round(r.sim_profile, 4),
                    "similarity_id_document": None if r.sim_id is None else round(r.sim_id, 4),
                    "threshold": t, "faces_in_frame": r.faces, "engine": self.engine.engine_name,
                    "face_box": None if r.box is None else dict(zip("xywh", r.box))}

        used: set[int] = set()

        def add(r: FrameIdentity, label: str, reason: str, jpeg: bytes | None) -> None:
            if jpeg is None or r.offset_ms in used:
                return
            used.add(r.offset_ms)
            evidence.append(EvidenceCandidate("IDENTITY", label, r.offset_ms, jpeg, r.sim_profile, model_result(r, reason)))

        by_offset = {r.offset_ms: r for r in records}
        for i, seg in enumerate(segments[:MAX_SEGMENT_EVIDENCE]):
            off = seg.pop("_lowest_offset")
            seg["evidence_offset_ms"] = off
            add(by_offset[off], "LOW_SIMILARITY_PERIOD", f"Lowest similarity in low-similarity period {i + 1}",
                self._thumbs.get(off))
        for seg in segments[MAX_SEGMENT_EVIDENCE:]:
            seg.pop("_lowest_offset", None)
        if self._worst:
            add(self._worst[0], "LOWEST_SIMILARITY", "Lowest similarity in the interview",
                self._annotate(*_rescaled(self._worst)))
        if self._best:
            add(self._best[0], "HIGHEST_SIMILARITY", "Highest similarity in the interview",
                self._annotate(*_rescaled(self._best)))
        kept = [r for r in face_frames if r.offset_ms in self._thumbs and r.offset_ms not in used]
        if kept:
            for k in range(REPRESENTATIVE_FRAMES):
                r = kept[min(len(kept) - 1, int((k + 0.5) * len(kept) / REPRESENTATIVE_FRAMES))]
                add(r, "SAMPLE_FRAME", "Evenly spaced sample frame", self._thumbs.get(r.offset_ms))

        summary = {
            "status": "completed",
            "engine": self.engine.engine_name,
            "threshold": t,
            "references": {"profile": True, "id_capture": id_capture_embedding is not None},
            "comparisons": comparisons,
            "overall": {"result": overall, "explanation": explanation},
            "video": {
                "frames_analyzed": len(records),
                "frames_with_face": len(face_frames),
                "frames_without_face": sum(1 for r in records if r.faces == 0),
                "frames_with_multiple_faces": sum(1 for r in records if r.faces > 1),
                "frames_failed": self.failed_frames,
            },
            "low_similarity_periods": segments,
            "timeline": timeline,
        }
        return summary, evidence
