"""Unit tests for decision logic - no database, no model files, no real personal data."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from app.ai.id_document import extract_dob, extract_id_number, extract_name, find_dates, match_person_name
from app.ai.ocr import TextLine
from app.core import security
from app.pipeline import frames as frame_source
from app.pipeline.environment import EnvironmentalDetector
from tests.conftest import synthetic_webm


def lines(*texts: str) -> list[TextLine]:
    return [TextLine(t, 0.95, (0, i * 20, 100, 18)) for i, t in enumerate(texts)]


# ------------------------------------------------------------------ security
def test_password_hash_is_argon2id_and_verifies():
    h = security.hash_password("Secret123")
    assert h.startswith("$argon2id$") and "Secret123" not in h
    assert security.verify_password(h, "Secret123")
    assert not security.verify_password(h, "secret123")
    assert not security.verify_password(None, "Secret123")


# ------------------------------------------------------------------ ID name / DOB matching
def test_name_matches_in_any_order_with_extra_words():
    m = match_person_name("Eswaradithya", "Palla", lines("INCOME TAX DEPARTMENT", "Palla Eswaradithya Yadav", "20/09/2003"))
    assert m.matched and m.score == 100
    assert m.order_same is False and m.extra_words == ["YADAV"]


def test_name_tolerates_ocr_spelling_noise():
    assert match_person_name("Eswaradithya", "Palla", lines("PALLA ESWARADITYA YADAV")).matched


def test_name_split_across_ocr_boxes_and_initials():
    assert match_person_name("Srinivasa Rao", "B", lines("DRIVING LICENCE", "SRINIVASARAO", "B", "BHASKARA RAO")).matched


def test_name_needs_every_part():
    # Same surname on the father's-name line is not a match for the candidate.
    assert not match_person_name("Eswaradithya", "Palla", lines("Palla Venkata Rao", "Eswar Kumar")).matched
    assert not match_person_name("Priya", "Sharma", lines("Government of India", "Medarametla Varun Chowdary")).matched


def test_dates_found_even_when_glued_to_labels():
    found = find_dates(lines("/DO15/08/2002", "Issued: 18/09/2012", "DOB 3 MAR 1999"))
    assert date(2002, 8, 15) in found and date(2012, 9, 18) in found and date(1999, 3, 3) in found


# ------------------------------------------------------------------ ID details shown to the candidate
def test_id_numbers_of_common_indian_ids():
    assert extract_id_number(lines("Government of India", "8969 5612 0414")) == ("Aadhaar", "8969 5612 0414")
    assert extract_id_number(lines("INCOME TAX DEPARTMENT", "Permanent Account Number Card", "DYAPG5625J")) == ("PAN", "DYAPG5625J")
    assert extract_id_number(lines("DL No: AP03 2011 5972009")) == ("Driving licence", "AP03 2011 5972009")
    assert extract_id_number(lines("Name Test")) == (None, None)


def test_name_and_dob_read_from_the_id():
    ocr = lines("GOVERNMENT OF INDIA", "Palla Eswaradithya Yadav", "DOB: 20/09/2003", "MALE")
    assert extract_name(ocr, "Eswaradithya", "Palla") == "Palla Eswaradithya Yadav"
    assert extract_dob(ocr) == "2003-09-20"
    assert extract_name(lines("INCOME TAX DEPARTMENT", "Name", "SAI GANESH"), "", "") == "SAI GANESH"
    assert extract_dob(lines("Year of Birth : 1999")) == "1999"


# ------------------------------------------------------------------ environment rules (Streamlit logic)
class _Boxes:
    def __init__(self, items):            # items: (class id, confidence)
        self.cls = [c for c, _ in items]
        self.conf = [f for _, f in items]
        self.xyxy = [(10, 10, 100, 100)] * len(items)

    def __len__(self):
        return len(self.cls)


class _Result:
    def __init__(self, *items):
        self.boxes = _Boxes(list(items))


NAMES = {0: "person", 67: "cell phone", 63: "laptop", 62: "tv", 73: "book"}
PERSON, PHONE_CLS, LAPTOP_CLS = 0, 67, 63


def _frame(value=120):
    f = np.full((120, 160, 3), value, np.uint8)
    f[::2, ::2] = 255 - value      # texture, so the frame is not "covered"
    return f


def _run(detector, *results, frame=None):
    out = []
    for r in results:
        out.append([a["type"] for a in detector.detect((frame if frame is not None else _frame()).copy(), r)[1]])
    return out


def test_second_and_multiple_persons():
    d = EnvironmentalDetector(NAMES)
    assert _run(d, _Result((PERSON, 0.9)), _Result((PERSON, 0.9), (PERSON, 0.6)), _Result((PERSON, 0.9), (PERSON, 0.6), (PERSON, 0.7)),
                _Result((PERSON, 0.9), (PERSON, 0.4))) == [[], ["Second Person Detected"], ["Multiple Persons Detected"], []]


def test_phone_needs_one_frame_and_laptop_two_consecutive_frames():
    d = EnvironmentalDetector(NAMES)
    seen = _run(d, _Result((PERSON, 0.9), (PHONE_CLS, 0.5)), _Result((PERSON, 0.9), (PHONE_CLS, 0.4)),
                _Result((PERSON, 0.9), (LAPTOP_CLS, 0.45)), _Result((PERSON, 0.9)), _Result((PERSON, 0.9), (PHONE_CLS, 0.3)),
                _Result((PERSON, 0.9), (LAPTOP_CLS, 0.45)), _Result((PERSON, 0.9), (LAPTOP_CLS, 0.5)))
    assert seen == [["Mobile Phone Detected"], ["Mobile Phone Detected"], [], [], [], [], ["Laptop Detected"]]


def test_person_left_screen_after_five_frames_and_camera_covered():
    d = EnvironmentalDetector(NAMES)
    seen = _run(d, *[_Result() for _ in range(6)])
    assert seen[4] == ["Person Left Screen"] and seen[5] == []
    d = EnvironmentalDetector(NAMES)
    dark = np.zeros((120, 160, 3), np.uint8)
    assert _run(d, *[_Result((PERSON, 0.9)) for _ in range(3)], frame=dark)[2] == ["Camera Covered"]


# ------------------------------------------------------------------ frame extraction
def test_frame_sampling_and_neighbour_lookup(tmp_path: Path):
    video = tmp_path / "v.webm"
    duration = synthetic_webm(video, seconds=5, fps=10)
    sampled = list(frame_source.sample_frames(video, 1000))
    assert len(sampled) == pytest.approx(duration / 1000, abs=1)
    assert [t for t, _ in sampled][:3] == [0, 1000, 2000]
    found = frame_source.frames_at(video, [1400, 2600])
    assert set(found) == {1400, 2600} and abs(found[1400][0] - 1400) <= 120


def test_growing_file_waits_for_data_until_finished(tmp_path: Path):
    import threading

    from app.pipeline.live import GrowingFile, LiveStopped

    path = tmp_path / "rec.webm"
    finished, stop = threading.Event(), threading.Event()
    reader = GrowingFile(path, finished, stop, idle_timeout_s=5)

    def upload():
        path.write_bytes(b"abc")
        finished.set()

    threading.Timer(0.3, upload).start()
    assert reader.read(10) == b"abc"            # waited for the file to appear
    assert reader.read(10) == b""               # finished and nothing left: end of stream
    reader.close()

    quiet = GrowingFile(tmp_path / "never.webm", threading.Event(), threading.Event(), idle_timeout_s=0.5)
    with pytest.raises(LiveStopped):
        quiet.read(10)                          # no upload at all: gives up after the idle timeout


def test_live_frames_match_the_full_sweep(tmp_path: Path):
    import threading

    from app.pipeline.live import GrowingFile, sample_growing_frames

    video = tmp_path / "full.webm"
    synthetic_webm(video, seconds=4.0)
    data = video.read_bytes()
    growing = tmp_path / "growing.webm"
    finished = threading.Event()

    def upload():
        with open(growing, "ab") as f:
            for i in range(0, len(data), 4096):
                f.write(data[i:i + 4096])
                f.flush()
        finished.set()

    threading.Thread(target=upload, daemon=True).start()
    info: dict = {}
    live = [ts for ts, _ in sample_growing_frames(GrowingFile(growing, finished, threading.Event(), 30), 1000, info)]
    assert live == [ts for ts, _ in frame_source.sample_frames(video, 1000)]
    assert info["width"] > 0


def test_unreadable_video_raises(tmp_path: Path):
    bad = tmp_path / "bad.webm"
    bad.write_bytes(b"\x1a\x45\xdf\xa3 not really a video")
    with pytest.raises(frame_source.FrameExtractionError):
        frame_source.probe(bad)


def test_id_photo_similarity_has_an_inconclusive_band():
    from app.pipeline.identity import id_photo_label
    # Measured: genuine printed-ID vs person 0.25-0.97, different people <= 0.21.
    assert id_photo_label(0.35) == "CONSISTENT"
    assert id_photo_label(0.25) == "INCONCLUSIVE"
    assert id_photo_label(0.12) == "NOT_CONSISTENT"
    assert id_photo_label(None) == "UNAVAILABLE"


def test_recruiter_email_has_autofill_link_and_goes_to_recruiter(monkeypatch):
    from datetime import datetime
    from urllib.parse import parse_qs, urlsplit

    from app.core.config import get_settings
    from app.services import mail_service, recruiter_service

    access = recruiter_service.IssuedAccess(7, "rec-ab12cd34@trustpro.local", "Ab3d-Ef5h-Jk7m", datetime(2026, 10, 15),
                                            "https://ui.example/recruiter/login", "Ravi Kumar")
    link = urlsplit(access.autofill_url)
    assert link.query == "" and parse_qs(link.fragment) == {"login": [access.login_id], "password": [access.password]}

    sent = {}
    monkeypatch.setattr(mail_service, "send_mail", lambda to, subject, html, text: sent.update(to=to, html=html) or True)
    monkeypatch.setattr(get_settings(), "recruiter_email", "")
    assert recruiter_service.email_access(access) is False and not sent        # no address: nothing sent
    monkeypatch.setattr(get_settings(), "recruiter_email", "hr@example.com, lead@example.com")
    assert recruiter_service.email_access(access) is True
    assert sent["to"] == ["hr@example.com", " lead@example.com"] and "Generate Report" in sent["html"]
