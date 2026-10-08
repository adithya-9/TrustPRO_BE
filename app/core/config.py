"""Application settings, read from environment variables (and an optional .env file)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ---------------- Application ----------------
    app_name: str = "TrustPRO"
    environment: str = "development"
    log_level: str = "INFO"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])

    # ---------------- Database ----------------
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/trustmate"
    db_schema: str = "epsoft"
    db_pool_size: int = 10

    # ---------------- Storage ----------------
    storage_dir: Path = BACKEND_ROOT / "storage"
    models_dir: Path = BACKEND_ROOT / "models"
    # Download missing model files at start-up (scripts/download_models.py). Needed on hosts
    # whose disk starts empty, such as Hugging Face Spaces.
    auto_download_models: bool = True
    # 4 MB: the hosted UI proxies uploads through Vercel, which limits request bodies to ~4.5 MB.
    max_image_bytes: int = 4 * 1024 * 1024
    max_video_bytes: int = 2 * 1024 * 1024 * 1024
    max_video_chunk_bytes: int = 16 * 1024 * 1024

    # ---------------- Sessions / secrets ----------------
    session_cookie_name: str = "trustpro_session"
    session_ttl_hours: int = 12
    cookie_secure: bool = False
    max_failed_logins: int = 5

    # ---------------- Recruiter report access ----------------
    # A separate cookie, so a candidate and a recruiter can be signed in in the same browser.
    recruiter_cookie_name: str = "trustpro_recruiter_session"
    recruiter_access_days: int = 7
    # Base URL of the web app put into the recruiter link, e.g. a tunnel URL when sharing outside
    # the office network. Empty = this machine's network address on the UI port.
    public_app_url: str = ""
    ui_port: int = 5173
    # Where the recruiter login for each finished report is emailed (comma-separated). Empty = only
    # printed in the server terminal.
    recruiter_email: str = ""

    # ---------------- Outgoing email (same providers as tl-core: AZURE | SENDGRID, plus SMTP) ----
    mail_provider: str = "none"          # azure | sendgrid | smtp | none
    mail_from: str = ""                  # sender address (an Azure/Office 365 mailbox for azure)
    azure_tenant_id: str = ""
    azure_client_id: str = ""
    azure_client_secret: str = ""
    azure_scope: str = "https://graph.microsoft.com/.default"
    azure_token_url: str = "https://login.microsoftonline.com/{tenantId}/oauth2/v2.0/token"
    azure_graph_url: str = "https://graph.microsoft.com/v1.0/users/{fromEmail}/sendMail"
    sendgrid_api_key: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_starttls: bool = True

    # ---------------- AI: general ----------------
    # ONNX Runtime execution providers in priority order. CPU is always appended as fallback,
    # so "CUDAExecutionProvider" can be added later without code changes.
    onnx_providers: list[str] = Field(default_factory=lambda: ["CPUExecutionProvider"])
    inference_threads: int = 3

    # ---------------- AI: face ----------------
    face_detect_score: float = 0.80
    # "arcface" (best separation, non-commercial weights) or "sface" (Apache-2.0). See docs/MODEL_SELECTION.md.
    face_engine: str = "arcface"
    arcface_match_threshold: float = 0.40
    # Printed ID photos are small, flat and often years old, so genuine ID-vs-person similarity is
    # lower than live-vs-live. Measured: genuine 0.25-0.97, different people <= 0.21. Between the
    # floor and the threshold the result is "inconclusive" (human review), not "not consistent".
    id_photo_match_threshold: float = 0.30
    id_photo_uncertain_floor: float = 0.20
    # SFace threshold recommended by the OpenCV Zoo authors (LFW-calibrated).
    sface_match_threshold: float = 0.363

    # ---------------- AI: environment (recorded video) ----------------
    # Same model and rules as the Environment_anomalies_streamlit project (Ultralytics YOLO11m);
    # thresholds are in app/pipeline/environment.py.
    # The OpenVINO export of the same model (scripts/export_environment_model.py) runs faster on
    # Intel CPUs with the same results; the .pt file is used when the export is missing.
    environment_model: str = "yolo11m_openvino_model"
    environment_model_fallback: str = "yolo11m.pt"

    # ---------------- Post-interview analysis ----------------
    analysis_sample_interval_s: float = 3.0
    # Faces are compared on one sampled frame every this many seconds (identity does not change
    # second to second); the environment check still sees every sampled frame.
    analysis_face_interval_s: float = 5.0
    analysis_identity_max_frames: int = 600
    analysis_job_workers: int = 1
    # Analyse the recording while it is uploaded, so the report is ready right after the
    # assessment (app/pipeline/live.py). Off = analyse the whole recording after it ends.
    live_analysis_enabled: bool = True
    # A recording that receives no new data for this long stops being followed.
    live_analysis_idle_timeout_s: float = 600.0

    @field_validator("cors_origins", "onnx_providers", mode="before")
    @classmethod
    def _split_csv(cls, value):
        if isinstance(value, str) and not value.strip().startswith("["):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @field_validator("storage_dir", "models_dir", mode="after")
    @classmethod
    def _absolute(cls, value: Path) -> Path:
        return value if value.is_absolute() else (BACKEND_ROOT / value).resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
