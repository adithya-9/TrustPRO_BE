# TrustPRO backend

**TrustPRO - AI Interview Protector**, a Trustmate product. Python API for candidate
onboarding, government-ID capture, live gaze and environment monitoring during the interview,
and evidence-based post-interview reports in which the ID is checked against the profile.

All AI runs locally on CPU. No GPU and no cloud AI service are required.

The UI lives in `../trustpro_ui`.

## Requirements

- Windows, macOS or Linux with **Python 3.11**
- **PostgreSQL** with the existing `trustmate` database (TrustPRO uses the `epsoft` schema)
- About 400 MB of disk for model files, and storage space for recordings (about 9 MB per interview minute)

## Setup (Windows PowerShell)

```powershell
cd E:\trustumate\trustpro_backend
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install --force-reinstall --no-deps opencv-contrib-python==5.0.0.93   # see note in requirements.txt

copy .env.example .env          # then edit DATABASE_URL
python -m scripts.download_models   # MediaPipe, Intel gaze/head-pose, YuNet, ArcFace, D-FINE, PP-OCRv5
alembic upgrade head                # creates / updates the TrustPRO tables in trustmate.epsoft
python -m scripts.check_setup       # verifies DB, migrations, models, OpenCV and OpenVINO
```

RF-DETR Medium (the environment model) is exported once from Roboflow's official package. The
export needs PyTorch, so it runs in a separate, throw-away environment (a short path avoids
Windows' path-length limit):

```powershell
py -3.11 -m venv $env:TEMP\rfx
& "$env:TEMP\rfx\Scripts\pip" install torch torchvision --index-url https://download.pytorch.org/whl/cpu
& "$env:TEMP\rfx\Scripts\pip" install "rfdetr[onnx]"
& "$env:TEMP\rfx\Scripts\python" scripts\export_rfdetr.py models
```

## Run

```powershell
uvicorn app.main:app --port 8000
```

- API docs: http://localhost:8000/api/docs
- Health: http://localhost:8000/api/health

There is no separate AI worker to start. Report generation runs in a background thread pool
inside the API process, with job state stored in `epsoft.interview_reports`, and unfinished
jobs resume after a restart. Live monitoring runs over the `/api/interviews/{id}/monitor`
WebSocket.

## Tests

```powershell
pytest
```

The suite has 40 tests and runs in about 30 s:
- `tests/test_units.py` covers:
  - password hashing,
  - order-independent name matching ("Eswaradithya Palla" vs "Palla Eswaradithya Yadav"),
  - dates glued to labels by OCR,
  - gaze calibration, classification, blinks and hysteresis,
  - the relevance rules (earbud-sized phones, background people, distant laptops,
    hand-as-person),
  - live and neighbour-frame confirmation,
  - the ID-photo "inconclusive" band,
  - WebM frame sampling.
- `tests/test_api.py` runs against the real `trustmate` database with throwaway users, which
  are deleted afterwards. It covers register/login/logout, lockout, validation, uploads
  rejected for type or missing face, ID capture that does not block, interview gating, the
  live monitor WebSocket (countdown calibration, then live gaze persistence), chunked upload
  ordering and retries, report generation with the ID check, partial failure handling, and
  access control between users.

Full journey with real media, without a browser:

```powershell
python -m scripts.simulate_interview --video interview.mp4 --profile-from-video-at 3 `
  --id-capture id.jpg --first-name Eswaradithya --last-name Palla --dob 1998-05-04
```

## Architecture

```
app/
  api/routes/      HTTP + WebSocket endpoints (thin: validation, auth, delegation)
  services/        business logic: auth, candidate profile, ID verification, interviews, reports, storage
  db/              SQLAlchemy models (mirror of the migration) and sessions
  ai/              model wrappers: face (YuNet + ArcFace/SFace), gaze (MediaPipe + Intel OpenVINO
                   gaze/head-pose), objects (RF-DETR-M on OpenVINO, D-FINE fallback), ocr,
                   id_document (name / DOB / photo checks)
  live/monitor.py  live session: per-frame gaze, throttled environment checks, event persistence
  pipeline/        post-interview analysis: frame extraction, identity + environment analyzers,
                   neighbour confirmation, gaze summary, job runner
  core/            settings, logging, security (Argon2id, session tokens, ID-number HMAC), errors
migrations/        Alembic (plain SQL, EPSoft conventions)
scripts/           download_models, check_setup, simulate_interview
docs/              MODEL_SELECTION.md, DATABASE.md, API.md
```

Post-interview analysis:

```
recording.webm --> frames (1 fps) --+--> IdentityAnalyzer (YuNet + ArcFace) -------------------+
                                    +--> EnvironmentAnalyzer (RF-DETR-M, parallel OpenVINO) -----+
ID captured before the interview ------> ID document: OCR name + DOB vs profile, photo vs profile +
                                                                                                 v
   neighbour-frame confirmation, ID photo vs every video face, gaze summary (live events) --> report
```

## Security notes

- Passwords: Argon2id. Sessions: random 256-bit tokens; only their SHA-256 is stored; HttpOnly cookie.
- Government ID numbers are not collected; the ID image is stored only as an access-controlled capture.
- Uploads: size limits, magic-byte type checks, full decode, and re-encoding to JPEG (removes EXIF/GPS and trailing data).
- Storage keys are generated server-side and confined to `STORAGE_DIR`; API responses only contain `/api/media/{id}` URLs, with ownership checked on every request.
- All SQL goes through SQLAlchemy with bound parameters.
- Logs contain ids only. No names, ID numbers or OCR text are logged.

## Configuration

All settings are environment variables (or `.env`); see `.env.example`. The important ones:

| Variable | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | - | `postgresql+psycopg://user:pass@host:5432/trustmate` |
| `STORAGE_DIR`, `MODELS_DIR` | `storage`, `models` | File locations |
| `FACE_ENGINE` | `arcface` | `arcface` (best accuracy, **non-commercial weights**) or `sface` (Apache-2.0) |
| `OBJECT_MODEL` | `rfdetr_m` | `rfdetr_m` (RF-DETR Medium) or `dfine_s` (faster fallback) |
| `GAZE_HORIZONTAL_DEG` / `GAZE_UP_DEG` / `GAZE_DOWN_SCORE` | 22 / 12 / 0.9 | Gaze thresholds relative to the countdown calibration |
| `ONNX_PROVIDERS` | `CPUExecutionProvider` | Add `CUDAExecutionProvider` with `onnxruntime-gpu` to use a GPU |
| `ANALYSIS_SAMPLE_INTERVAL_S` | `1.0` | Baseline sampling for reports |
| `LIVE_ENVIRONMENT_INTERVAL_S` | `1.0` | Live object-detection rate per interview |

## Known limitations

- The ArcFace weights are licensed for non-commercial research. Switch to `FACE_ENGINE=sface` or license InsightFace before commercial deployment (see docs/MODEL_SELECTION.md).
- Report generation takes roughly 1.1-1.5x the interview length on a 4-core laptop CPU; RF-DETR-M is the main cost.
- Live environment warnings check every 1.5 s and need two hits in a row, so very brief objects may only appear in the report.
- If the camera disconnects mid-interview, the interview can be ended with the recording so far. Recording cannot resume into the same file.
- A single API process is assumed. To scale out, move `run_report` to a worker process and route live-monitor WebSockets by interview.
