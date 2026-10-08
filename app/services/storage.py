"""File storage for photos, ID images, recordings and evidence frames.

Bytes live on the local filesystem under STORAGE_DIR; PostgreSQL keeps only metadata
(epsoft.media_files). Storage keys are generated here from ids and random tokens - never
from user input - and every resolved path is checked to stay inside the storage root,
so path traversal is not possible. Swapping this class for an S3/Azure Blob implementation
only needs the same four methods.
"""
from __future__ import annotations

import hashlib
import secrets
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.errors import InvalidUpload, NotFound
from app.db.models import MediaFile

IMAGE_CONTENT_TYPES = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}


class LocalStorage:
    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or get_settings().storage_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if self.root not in path.parents:
            raise NotFound("File not found.")
        return path

    @staticmethod
    def new_key(folder: str, extension: str) -> str:
        return f"{folder}/{secrets.token_hex(16)}{extension}"

    def write(self, key: str, data: bytes) -> Path:
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(path)
        return path

    def append(self, key: str, data: bytes) -> int:
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "ab") as fh:
            fh.write(data)
            return fh.tell()

    def delete(self, key: str) -> None:
        try:
            self.path_for(key).unlink(missing_ok=True)
        except NotFound:
            pass


_storage: LocalStorage | None = None


def get_storage() -> LocalStorage:
    global _storage
    if _storage is None:
        _storage = LocalStorage()
    return _storage


# --------------------------------------------------------------------- images
def sniff_image_type(data: bytes) -> str | None:
    """Content type from magic bytes - the client's declared type is not trusted."""
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def decode_image(data: bytes, label: str) -> np.ndarray:
    settings = get_settings()
    if not data:
        raise InvalidUpload(f"The {label} is empty. Please choose an image file.")
    if len(data) > settings.max_image_bytes:
        raise InvalidUpload(f"The {label} is too large. The maximum size is {settings.max_image_bytes // (1024 * 1024)} MB.")
    if sniff_image_type(data) is None:
        raise InvalidUpload(f"The {label} must be a JPEG, PNG or WebP image.")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidUpload(f"The {label} could not be read. The file may be damaged.")
    h, w = image.shape[:2]
    if min(h, w) < 200:
        raise InvalidUpload(f"The {label} is too small. Please use an image at least 200 pixels on each side.")
    if max(h, w) > 8000:
        raise InvalidUpload(f"The {label} resolution is too high. Please use an image under 8000 pixels.")
    return image


def encode_jpeg(image: np.ndarray, quality: int = 88, max_side: int | None = None) -> bytes:
    if max_side:
        h, w = image.shape[:2]
        scale = max_side / max(h, w)
        if scale < 1.0:
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buffer.tobytes()


def store_image(db: Session, owner_user_id: int, category: str, folder: str, image: np.ndarray,
                *, quality: int = 88, max_side: int | None = 1600) -> MediaFile:
    """Re-encode to JPEG (strips EXIF/GPS metadata and any trailing payload) and record it."""
    if max_side:
        h, w = image.shape[:2]
        scale = max_side / max(h, w)
        if scale < 1.0:
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    data = encode_jpeg(image, quality=quality)
    return store_bytes(db, owner_user_id, category, folder, data, "image/jpeg", ".jpg",
                       width=image.shape[1], height=image.shape[0])


def store_bytes(db: Session, owner_user_id: int, category: str, folder: str, data: bytes,
                content_type: str, extension: str, *, width: int | None = None,
                height: int | None = None) -> MediaFile:
    storage = get_storage()
    key = storage.new_key(folder, extension)
    storage.write(key, data)
    media = MediaFile(
        owner_user_id=owner_user_id, file_category=category, storage_key=key, content_type=content_type,
        size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), width=width, height=height,
    )
    db.add(media)
    db.flush()
    return media


def read_image(media: MediaFile) -> np.ndarray:
    path = get_storage().path_for(media.storage_key)
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise NotFound("Stored image could not be read.")
    return image
