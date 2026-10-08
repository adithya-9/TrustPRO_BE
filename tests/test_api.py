"""API integration tests against the real database (see conftest.py)."""
from __future__ import annotations

import time
from datetime import timedelta

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.core.config import get_settings
from app.db.models import Interview, InterviewReport, ReportAccess, User, utcnow
from app.main import app
from app.services import recruiter_service
from app.services.storage import get_storage
from tests.conftest import PASSWORD, jpeg_bytes, synthetic_image, synthetic_webm


# ------------------------------------------------------------------ authentication
def test_register_login_logout_flow(make_user):
    c, email = make_user()
    me = c.get("/api/auth/me").json()
    assert me["email"] == email and me["next_step"] == "PROFILE"

    assert c.post("/api/auth/logout").status_code == 200
    assert c.get("/api/auth/me").status_code == 401

    r = c.post("/api/auth/login", json={"email": email.upper(), "password": PASSWORD})
    assert r.status_code == 200 and r.json()["email"] == email
    assert "httponly" in r.headers["set-cookie"].lower()
    assert c.get("/api/auth/me").status_code == 200


def test_password_is_not_stored_in_plaintext(make_user, db):
    _, email = make_user()
    user = db.scalar(select(User).where(User.email == email))
    assert PASSWORD not in user.password_hash and user.password_hash.startswith("$argon2id$")


def test_duplicate_registration_and_validation(make_user, client):
    _, email = make_user()
    r = client.post("/api/auth/register", json={"email": email, "password": PASSWORD})
    assert r.status_code == 409 and r.json()["error"]["code"] == "EMAIL_TAKEN"
    r = client.post("/api/auth/register", json={"email": "not-an-email", "password": "abc"})
    assert r.status_code == 422 and set(r.json()["error"]["details"]["fields"]) == {"email", "password"}


def test_wrong_password_and_lockout(make_user):
    c, email = make_user(signed_in=False)
    for _ in range(5):
        r = c.post("/api/auth/login", json={"email": email, "password": "Wrongpass1"})
        assert r.status_code == 401 and r.json()["error"]["code"] == "INVALID_CREDENTIALS"
    r = c.post("/api/auth/login", json={"email": email, "password": PASSWORD})
    assert r.status_code == 429 and r.json()["error"]["code"] == "ACCOUNT_LOCKED"


def test_protected_endpoints_require_session(client):
    fresh = client.__class__(client.app)
    for method, url in [("get", "/api/candidate/profile"), ("get", "/api/interviews"), ("post", "/api/interviews")]:
        assert getattr(fresh, method)(url).status_code == 401


# ------------------------------------------------------------------ profile
PROFILE = {"first_name": "Eswaradithya", "last_name": "Palla", "date_of_birth": "1998-05-04",
           "mobile_number": "9876543210", "city": "Hyderabad"}


def test_profile_requires_photo_and_rejects_non_images(make_user):
    c, _ = make_user()
    r = c.put("/api/candidate/profile", data=PROFILE)
    assert r.status_code == 422 and r.json()["error"]["code"] == "PROFILE_PHOTO_REQUIRED"
    r = c.put("/api/candidate/profile", data=PROFILE,
              files={"profile_photo": ("x.jpg", b"MZ\x90\x00 definitely not an image", "image/jpeg")})
    assert r.status_code == 422 and "JPEG, PNG or WebP" in r.json()["error"]["message"]


def test_profile_photo_without_a_face_is_rejected(make_user):
    c, _ = make_user()
    r = c.put("/api/candidate/profile", data=PROFILE,
              files={"profile_photo": ("p.jpg", jpeg_bytes(synthetic_image(seed=3)), "image/jpeg")})
    assert r.status_code == 422 and r.json()["error"]["code"] == "PROFILE_PHOTO_NO_FACE"


def test_profile_field_validation(make_user):
    c, _ = make_user()
    r = c.put("/api/candidate/profile", data={**PROFILE, "mobile_number": "12ab", "first_name": "J0hn"})
    assert r.status_code == 422
    assert set(r.json()["error"]["details"]["fields"]) & {"mobile_number", "first_name"}


def test_id_capture_requires_profile(make_user):
    c, _ = make_user()
    r = c.post("/api/candidate/id-verifications", files={"capture": ("c.jpg", jpeg_bytes(synthetic_image()), "image/jpeg")})
    assert r.status_code == 409 and r.json()["error"]["code"] == "PROFILE_INCOMPLETE"


def test_id_capture_is_read_then_saved_by_the_candidate(verified_candidate):
    c = verified_candidate
    r = c.post("/api/candidate/id-verifications", files={"capture": ("c.jpg", jpeg_bytes(synthetic_image(seed=9)), "image/jpeg")})
    assert r.status_code == 201
    body = r.json()
    assert body["attempt_no"] == 2 and body["capture_url"].startswith("/api/media/") and body["confirmed"] is False
    assert set(body["details"]) == {"name", "id_number", "id_type", "dob"}
    assert c.get("/api/auth/me").json()["next_step"] == "ID_VERIFICATION"         # not saved yet
    r = c.post("/api/interviews")
    assert r.status_code == 409 and r.json()["error"]["code"] == "ID_NOT_CONFIRMED"
    saved = c.post(f"/api/candidate/id-verifications/{body['verification_id']}/confirm").json()
    assert saved["confirmed"] is True
    assert c.get("/api/auth/me").json()["next_step"] == "INTERVIEW"
    assert c.post("/api/interviews").status_code == 201


# ------------------------------------------------------------------ interviews
def test_interview_requires_id_capture(make_user):
    c, _ = make_user()
    r = c.post("/api/interviews")
    assert r.status_code == 409


def test_interview_lifecycle_recording_and_report(verified_candidate, tmp_path, db):
    c = verified_candidate
    interview = c.post("/api/interviews").json()
    iid = interview["interview_id"]
    assert interview["status"] == "CREATED"
    assert c.post(f"/api/interviews/{iid}/start").json()["status"] == "LIVE"

    # ---- chunked upload: order enforced, retries idempotent, non-WebM rejected
    video = tmp_path / "rec.webm"
    # Long enough for 6 sampled frames at the configured interval ("Person Left Screen" needs 5).
    duration = synthetic_webm(video, seconds=6 * max(1.0, get_settings().analysis_sample_interval_s))
    blob = video.read_bytes()
    parts = [blob[i:i + 20000] for i in range(0, len(blob), 20000)]
    bad = c.put(f"/api/interviews/{iid}/recording/chunks/0", content=b"not webm data")
    assert bad.status_code == 422
    assert c.put(f"/api/interviews/{iid}/recording/chunks/1", content=parts[0]).status_code == 409
    for seq, part in enumerate(parts):
        r = c.put(f"/api/interviews/{iid}/recording/chunks/{seq}", content=part)
        assert r.status_code == 200 and r.json()["next_seq"] == seq + 1
    assert c.put(f"/api/interviews/{iid}/recording/chunks/0", content=parts[0]).json()["next_seq"] == len(parts)

    r = c.post(f"/api/interviews/{iid}/end", json={"duration_ms": duration, "chunk_count": len(parts) + 1})
    assert r.status_code == 409 and r.json()["error"]["code"] == "RECORDING_INCOMPLETE"
    ended = c.post(f"/api/interviews/{iid}/end", json={"duration_ms": duration, "chunk_count": len(parts)}).json()
    assert ended["status"] == "ENDED" and ended["recording_status"] == "COMPLETE"
    assert c.get(ended["recording_url"]).content == blob

    # ---- report: queued automatically when the assessment ends, written as an HTML file.
    report = None
    deadline = time.time() + 300
    while time.time() < deadline:
        db.expire_all()
        report = db.scalar(select(InterviewReport).where(InterviewReport.interview_id == iid))
        if report and report.status in ("C", "F"):
            break
        time.sleep(1)
    assert report is not None and report.status == "C", report and report.error_message
    assert report.processing_info["timings"]["mode"] == "live"      # analysed while the recording was uploaded
    assert report.environment_summary["status"] == "completed" and report.environment_summary["frames_analyzed"] >= 5
    assert "Person Left Screen" in report.environment_summary["events_by_type"]   # nobody in the synthetic video
    rec = db.get(Interview, iid).recording
    html = get_storage().path_for(rec.storage_key).parent / f"report_{report.report_id}.html"
    text = html.read_text(encoding="utf-8")
    assert "ID verification" in text and "Environmental anomalies" in text and "Person Left Screen" in text
    assert "candidates" in str(html) and "assessments" in str(html)

    # ---- recruiter access: a login for this report only is created when the report completes.
    auto = None
    while time.time() < deadline and auto is None:
        db.expire_all()
        auto = db.scalar(select(ReportAccess).where(ReportAccess.report_id == report.report_id))
        time.sleep(0.2)
    assert auto is not None and db.get(User, auto.user_id).user_type == "R"
    old_login = auto.user_id
    issued = recruiter_service.issue_access(db, report.report_id)        # replaces it: the old login is gone
    db.expire_all()
    assert db.get(User, old_login) is None
    assert issued.login_id.startswith("rec-") and issued.login_url.endswith("/recruiter/login")

    rc = TestClient(app)
    rc.portal = c.portal
    assert rc.post("/api/recruiter/login", json={"email": issued.login_id, "password": "wrong-pass1"}).status_code == 401
    email = c.get("/api/auth/me").json()["email"]
    assert c.post("/api/recruiter/login", json={"email": email, "password": PASSWORD}).status_code == 401  # candidate
    assert rc.post("/api/auth/login", json={"email": issued.login_id, "password": issued.password}).status_code == 401
    me = rc.post("/api/recruiter/login", json={"email": issued.login_id, "password": issued.password})
    assert me.status_code == 200 and me.json()["report_id"] == report.report_id
    page = rc.get("/api/recruiter/report")
    assert page.status_code == 200 and "Environmental anomalies" in page.text
    assert "default-src 'none'" in page.headers["content-security-policy"]
    assert rc.get("/api/auth/me").status_code == 401                     # no access to candidate pages
    assert rc.get(f"/api/interviews/{iid}").status_code == 401
    assert c.get("/api/recruiter/report").status_code == 401              # candidate cannot open it

    access = db.scalar(select(ReportAccess).where(ReportAccess.report_id == report.report_id))
    access.expires_at = utcnow() - timedelta(minutes=1)
    db.commit()
    r = rc.get("/api/recruiter/report")
    assert r.status_code == 401 and r.json()["error"]["code"] == "ACCESS_EXPIRED"
    assert rc.post("/api/recruiter/logout").status_code == 200


def test_other_users_cannot_access_interviews_or_media(verified_candidate, make_user):
    owner = verified_candidate
    iid = owner.post("/api/interviews").json()["interview_id"]
    photo_url = owner.get("/api/candidate/profile").json()["profile_photo_url"]
    intruder, _ = make_user()
    assert intruder.get(f"/api/interviews/{iid}").status_code == 403
    assert intruder.get(photo_url).status_code == 403
    assert intruder.post(f"/api/interviews/{iid}/start").status_code == 403
    assert owner.get(photo_url).status_code == 200


def test_missing_interview_is_404(verified_candidate):
    assert verified_candidate.get("/api/interviews/999999999").status_code == 404


def test_media_and_recording_paths_are_not_exposed(verified_candidate):
    profile = verified_candidate.get("/api/candidate/profile").json()
    assert all("storage" not in str(v) and "\\" not in str(v) for v in profile.values())

