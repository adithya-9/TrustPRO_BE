# TrustPRO API

Interactive documentation (generated from the code): **http://localhost:8000/api/docs**
(ReDoc at `/api/redoc`, schema at `/api/openapi.json`).

## Conventions

- Authentication: `POST /api/auth/login` sets an **HttpOnly session cookie**
  (`trustpro_session`, SameSite=Lax). No JWT. Sessions are revocable server-side.
- The UI calls the API same-origin (Vite proxies `/api` in development).
- Every error has the same shape. `message` is written for the candidate:

```json
{ "error": { "code": "ID_NOT_CAPTURED", "message": "Please show your government ID on camera before starting the interview.", "details": {} } }
```

Validation errors use `code: "VALIDATION_ERROR"` with `details.fields` mapping field names to
messages. Database outages return `503 DATABASE_UNAVAILABLE`.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/auth/register` | Create a candidate account `{email, password}` and sign in |
| POST | `/api/auth/login` | Sign in. 401 `INVALID_CREDENTIALS`; 429 `ACCOUNT_LOCKED` after 5 failures (15 min) |
| POST | `/api/auth/logout` | Revoke the session |
| GET | `/api/auth/me` | User + `next_step` (`PROFILE` / `ID_VERIFICATION` / `INTERVIEW`) |
| GET | `/api/candidate/profile` | Profile |
| PUT | `/api/candidate/profile` | Multipart: `first_name`, `last_name`, `date_of_birth`, `mobile_number`, `city?`, `profile_photo`. The photo is required on first save and must show exactly one clear face. |
| POST | `/api/candidate/id-verifications` | Multipart `capture` (camera image of the ID). Stored immediately (201) with no checks at this step, so the candidate can start the interview. The ID is analysed during report generation. |
| GET | `/api/candidate/id-verifications/latest` | Most recent capture |
| GET | `/api/interviews` | Candidate's interviews (with latest report status) |
| POST | `/api/interviews` | Create. 409 `ID_NOT_CAPTURED` until an ID has been captured |
| GET | `/api/interviews/{id}` | Interview |
| POST | `/api/interviews/{id}/start` | Start, which opens the recording |
| PUT | `/api/interviews/{id}/recording/chunks/{seq}` | Raw WebM bytes (MediaRecorder timeslice). Sequential; repeats are acknowledged; gaps return 409 `CHUNK_OUT_OF_ORDER` with `expected_seq` |
| POST | `/api/interviews/{id}/end` | `{duration_ms, chunk_count}`. 409 `RECORDING_INCOMPLETE` if chunks are missing |
| GET | `/api/interviews/{id}/recording` | Stream the recording (HTTP range supported) |
| POST | `/api/interviews/{id}/reports` | Queue report generation (202). Returns immediately |
| GET | `/api/interviews/{id}/reports/latest` | Latest report status |
| GET | `/api/reports/{id}/status` | `QUEUED` / `PROCESSING` (with `progress_pct`, `current_stage`) / `COMPLETED` / `FAILED` (with `error_message`) |
| GET | `/api/reports/{id}` | Status, plus the full report once completed: `overview`, `identity`, `gaze`, `environment`, `evidence`, `processing` |
| GET | `/api/media/{file_id}` | Stored image (owner only). Storage paths are never exposed |
| GET | `/api/health` | Liveness + database check |

Ownership is enforced on every interview, report and media request (403 for other users).

## Live monitor WebSocket

`ws://<host>/api/interviews/{id}/monitor` uses the session cookie. The interview must be `LIVE`.
Close codes: `4401` not signed in, `4403` not allowed / not live.

There are two phases:
- **Calibration** is the 10-second countdown before recording. Frames build the candidate's
  neutral gaze reference; nothing is stored as gaze events.
- **Live** starts when the browser sends `{"type": "phase", "phase": "live"}` at the moment
  recording starts.

**Browser to server, binary:** an 8-byte little-endian float64 offset in ms followed by a JPEG
frame (about 640 px wide). The offset is countdown time during calibration and the recording
offset when live.

**Browser to server, text:** `{"type": "phase", "phase": "live"}`, and `end` to close the session
cleanly.

**Server to browser, JSON:**

```json
{"type": "ready", "calibrated": false}
{"type": "calibration", "frames": 14, "needed": 8, "face": true}
{"type": "calibrated", "ok": true, "frames": 42}
{"type": "gaze", "offset_ms": 12480, "direction": "LEFT", "stable_direction": "LEFT",
 "confidence": 0.82, "faces": 1, "calibrated": true}
{"type": "environment", "offset_ms": 12480, "available": true, "persons": 1,
 "active": [{"event_type": "PHONE", "title": "Mobile phone detected",
             "message": "Mobile phone detected. Please remove the phone from the interview area.",
             "confidence": 0.71}]}
{"type": "error", "code": "GAZE_FAILED", "message": "Gaze monitoring is temporarily unavailable."}
```

The browser sends the next frame only after the `gaze` reply, which provides back-pressure.
Gaze segments and environment events are stored server-side as they close. A reconnect
replaces the previous session for the same interview and reuses the stored calibration
(`interviews.gaze_calibration`).

## Report document (abridged)

```jsonc
{
  "status": "COMPLETED",
  "overview": { "candidate": { "first_name": "...", "last_name": "...", "date_of_birth": "...", "mobile_number": "...", "city": "...", "profile_photo_url": "..." }, "interview": {...} },
  "identity": {
    "status": "completed", "engine": "arcface", "threshold": 0.4,
    "id_document": { "status": "completed", "capture_url": "/api/media/88", "portrait_url": "/api/media/90",
                     "checks": { "portrait_detected": true, "text_readable": true,
                                 "name": { "matched": true, "score": 100, "text": "BOORADA SRINIVAS",
                                           "order_same": false, "extra_words": [] },
                                 "dob_match": true, "dates_found": ["1977-01-27"],
                                 "face_vs_profile": { "similarity": 0.25, "threshold": 0.3, "result": "INCONCLUSIVE" } } },
    "comparisons": [{ "pair": "PROFILE_ID", "similarity": 0.25, "result": "INCONCLUSIVE" },
                    { "pair": "PROFILE_VIDEO", "similarity": 0.86, "frames_compared": 112,
                      "consistent_pct": 99.1, "result": "CONSISTENT" },
                    { "pair": "ID_VIDEO", "...": "..." }],
    "overall": { "result": "CONSISTENT", "explanation": "In 100% of the 47 frames ..." },
    "low_similarity_periods": [...], "timeline": [[offset_ms, similarity, faces], ...],
    "evidence": [{ "label": "LOWEST_SIMILARITY", "offset_ms": 64013, "image_url": "/api/media/91",
                   "confidence": 0.318, "model_result": {...} }]
  },
  "gaze": { "by_direction": [...], "away_episodes_over_2s": 6, "events": [...] },
  "environment": { "events": [{ "event_type": "PHONE", "start_ms": 12000, "peak_confidence": 0.65,
                                "confirmed_by": "neighbouring_frames", "evidence": {...} }],
                   "live_events": [...] },
  "evidence": [...],
  "processing": { "sample_interval_ms": 1000, "frames_analyzed": 47, "timings": {...}, "models": {...} }
}
```

A section that could not be analysed reports `{"status": "failed", "error": "..."}` while the
rest of the report completes.
