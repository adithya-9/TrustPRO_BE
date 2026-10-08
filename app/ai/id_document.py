"""Reading the government ID captured on camera and comparing it with the candidate profile.

Runs during report generation. Every check produces an observable result (a score, a boolean
or "not found") that is shown in the report; nothing here labels a person.

Compared with the profile:
  * name - order-independent token matching (see ``match_person_name``),
  * date of birth - any date printed on the ID equal to the profile DOB,
  * photo - face similarity between the ID portrait and the profile photo.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

import cv2
import numpy as np
from rapidfuzz import fuzz

from app.ai.face import Face, crop_face, get_face_engine
from app.ai.ocr import TextLine, get_ocr_engine

# ------------------------------------------------------------------ thresholds
MIN_SHARPNESS = 60.0         # Laplacian variance on a 1000 px-wide greyscale image
MIN_BRIGHTNESS = 55.0
MAX_GLARE_FRACTION = 0.06    # share of near-white, unsaturated pixels
MIN_READABLE_LINES = 3
MIN_LINE_CONFIDENCE = 0.70
MIN_READABLE_CHARS = 15
NAME_TOKEN_MATCH = 85.0      # per-name-part similarity needed (0-100)
PORTRAIT_MIN_SCORE = 0.60    # printed ID photos are small and flat; YuNet scores them lower

DATE_PATTERNS = [
    # Digit lookarounds instead of \b: OCR often glues a label to the date ("DOB:" -> "DO15/08/2002").
    (re.compile(r"(?<!\d)(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})(?!\d)"), "dmy"),
    (re.compile(r"(?<!\d)(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})(?!\d)"), "ymd"),
]
MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}
MONTH_NAME_DATE = re.compile(r"(?<!\d)(\d{1,2})\s*([A-Z]{3})[A-Z]*\s*(\d{4})(?!\d)")


@dataclass
class DocumentQuality:
    sharpness: float
    brightness: float
    glare_fraction: float

    @property
    def blurry(self) -> bool:
        return self.sharpness < MIN_SHARPNESS

    @property
    def too_dark(self) -> bool:
        return self.brightness < MIN_BRIGHTNESS

    @property
    def glare(self) -> bool:
        return self.glare_fraction > MAX_GLARE_FRACTION

    def as_dict(self) -> dict:
        return {"sharpness": round(self.sharpness, 1), "brightness": round(self.brightness, 1),
                "glare_fraction": round(self.glare_fraction, 4), "blurry": self.blurry,
                "too_dark": self.too_dark, "glare": self.glare}


@dataclass
class NameMatch:
    score: float                       # 0-100, the weakest profile name part
    matched: bool
    text: str | None                   # the ID text the name was found in
    order_same: bool | None            # same word order as the profile
    extra_words: list[str] = field(default_factory=list)   # words on the ID not in the profile
    parts: dict[str, float] = field(default_factory=dict)  # per profile name part

    def as_dict(self) -> dict:
        return {"score": round(self.score, 1), "matched": self.matched, "text": self.text,
                "order_same": self.order_same, "extra_words": self.extra_words,
                "parts": {k: round(v, 1) for k, v in self.parts.items()}}


@dataclass
class DocumentReading:
    quality: DocumentQuality
    portrait: Face | None
    portrait_crop: np.ndarray | None
    lines: list[TextLine]
    readable_lines: int
    text_readable: bool
    name: NameMatch | None = None
    dates_found: list[str] = field(default_factory=list)
    dob_match: bool | None = None
    face_similarity_profile: float | None = None
    portrait_embedding: np.ndarray | None = None

    def extracted_fields(self) -> dict:
        return {
            "name": self.name.as_dict() if self.name else None,
            "dates_found": self.dates_found,
            "text_lines_read": len(self.lines),
            "text_lines_confident": self.readable_lines,
        }


# ------------------------------------------------------------------ quality
def measure_quality(image: np.ndarray) -> DocumentQuality:
    h, w = image.shape[:2]
    small = cv2.resize(image, (1000, max(1, int(h * 1000 / w))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    glare = np.logical_and(hsv[:, :, 2] >= 245, hsv[:, :, 1] <= 30).mean()
    return DocumentQuality(sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                           brightness=float(gray.mean()), glare_fraction=float(glare))


# ------------------------------------------------------------------ name matching
def name_tokens(value: str) -> list[str]:
    return [t for t in re.split(r"[^A-Z]+", value.upper()) if t]


def _part_score(part: str, words: list[str], compact: str) -> tuple[float, int | None]:
    """Best similarity of one profile name part against the ID words (0-100) and the index of
    the ID word it matched (None when matched inside merged OCR text)."""
    best, best_i = 0.0, None
    for i, word in enumerate(words):
        if len(part) == 1 or len(word) == 1:
            score = 100.0 if part[0] == word[0] else 0.0           # initials: "B" == "BHASKARA"
        else:
            score = float(fuzz.ratio(part, word))
        if score > best:
            best, best_i = score, i
    # OCR sometimes drops spaces ("SRINIVASARAO"): look for the part inside the merged text.
    if len(part) >= 4 and best < NAME_TOKEN_MATCH and compact:
        merged = float(fuzz.partial_ratio(part, compact)) * 0.98
        if merged > best:
            best, best_i = merged, None
    return best, best_i


def match_person_name(first_name: str, last_name: str, lines: list[TextLine]) -> NameMatch:
    """Find the profile name on the ID regardless of word order, allowing extra words.

    Example: profile "Eswaradithya Palla" matches ID text "PALLA ESWARADITHYA YADAV"
    (both parts found, order differs, extra word "YADAV").

    The ID text is searched in windows of up to three adjacent OCR lines (names wrap or are
    split into several boxes). Every profile name part must be found in the same window; the
    window's score is its weakest part, so one missing part means no match.
    """
    parts = name_tokens(first_name) + name_tokens(last_name)
    if not parts or not lines:
        return NameMatch(0.0, False, None, None)
    texts = [line.text for line in lines]
    best: tuple[float, int, int] | None = None
    best_detail = None
    for i in range(len(texts)):
        for span in (1, 2, 3):
            if i + span > len(texts):
                break
            window = " ".join(texts[i:i + span])
            words = name_tokens(window)
            if not words:
                continue
            compact = "".join(words)
            scored = [_part_score(p, words, compact) for p in parts]
            score = min(s for s, _ in scored)
            key = (score, -span, -len(words))      # prefer tighter windows with fewer words
            if best is None or key > best:
                best, best_detail = key, (window, words, scored)
    if best is None:
        return NameMatch(0.0, False, None, None)
    score = best[0]
    window, words, scored = best_detail
    matched = score >= NAME_TOKEN_MATCH
    positions = [i for _, i in scored if i is not None]
    order_same = positions == sorted(positions) if matched and len(positions) == len(parts) else None
    used = {i for _, i in scored if i is not None}
    extra = [w for i, w in enumerate(words) if i not in used] if matched and len(positions) == len(parts) else []
    return NameMatch(score=score, matched=matched, text=window.strip(), order_same=order_same,
                     extra_words=extra, parts={p: s for p, (s, _) in zip(parts, scored)})


# ------------------------------------------------------------------ dates
def find_dates(lines: list[TextLine]) -> list[date]:
    found: list[date] = []
    text = " ".join(line.text.upper() for line in lines)
    for pattern, order in DATE_PATTERNS:
        for m in pattern.finditer(text):
            a, b, c = (int(g) for g in m.groups())
            day, month, year = (a, b, c) if order == "dmy" else (c, b, a)
            try:
                found.append(date(year, month, day))
            except ValueError:
                continue
    for m in MONTH_NAME_DATE.finditer(text):
        month = MONTHS.get(m.group(2))
        if month:
            try:
                found.append(date(int(m.group(3)), month, int(m.group(1))))
            except ValueError:
                continue
    unique: list[date] = []
    for d in found:
        if d not in unique and 1900 <= d.year <= 2100:
            unique.append(d)
    return unique


# ------------------------------------------------------------------ details shown to the candidate
ID_NUMBER_PATTERNS = [
    ("Aadhaar", re.compile(r"(?<![\dX])([\dX]{4}\s?[\dX]{4}\s?\d{4})(?!\d)")),
    ("PAN", re.compile(r"(?<![A-Z0-9])([A-Z]{5}\d{4}[A-Z])(?![A-Z0-9])")),
    ("Driving licence", re.compile(r"(?<![A-Z0-9])([A-Z]{2}[\s-]?\d{2}[\s-]?\d{4}[\s-]?\d{7})(?!\d)")),
    ("Voter ID", re.compile(r"(?<![A-Z0-9])([A-Z]{3}\d{7})(?![A-Z0-9])")),
    ("Passport", re.compile(r"(?<![A-Z0-9])([A-Z]\d{7})(?![A-Z0-9])")),
]
_NAME_LABEL = re.compile(r"^\s*(NAME|NAAM|नाम)\s*[:/\-]?\s*", re.IGNORECASE)
_NOT_A_NAME = re.compile(r"GOVERNMENT|INDIA|INCOME|TAX|DEPARTMENT|LICEN|UNION|AUTHORITY|ELECTION|COMMISSION|"
                         r"FATHER|MOTHER|HUSBAND|ADDRESS|BIRTH|DOB|MALE|FEMALE|SIGNATURE|CARD|NUMBER|VALID|ISSUE|"
                         r"PERMANENT|ACCOUNT|AADHAAR|REPUBLIC|STATE|TRANSPORT|DATE|YEAR")
_DOB_LABEL = re.compile(r"DOB|D\.O\.B|BIRTH|जन्म", re.IGNORECASE)
_YEAR_OF_BIRTH = re.compile(r"(?:YEAR OF BIRTH|YOB)\D{0,5}((?:19|20)\d{2})", re.IGNORECASE)


def extract_id_number(lines: list[TextLine]) -> tuple[str | None, str | None]:
    """(ID type, number) of the first known Indian ID number format found."""
    text = " ".join(line.text.upper() for line in lines)
    for id_type, pattern in ID_NUMBER_PATTERNS:
        m = pattern.search(text)
        if m:
            return id_type, re.sub(r"\s+", " ", m.group(1).strip())
    return None, None


def extract_name(lines: list[TextLine], first_name: str, last_name: str) -> str | None:
    """The name as printed on the ID: the text matching the profile name, otherwise the line after
    (or on) a "Name" label, otherwise the first line that looks like a person's name."""
    match = match_person_name(first_name, last_name, lines) if (first_name or last_name) else None
    if match and match.matched and match.text:
        return _NAME_LABEL.sub("", match.text).strip()
    for i, line in enumerate(lines):
        if _NAME_LABEL.match(line.text):
            rest = _NAME_LABEL.sub("", line.text).strip()
            if len(rest) >= 3:
                return rest
            if i + 1 < len(lines):
                return lines[i + 1].text.strip()
    for line in lines:
        words = name_tokens(line.text)
        if 2 <= len(words) <= 4 and not _NOT_A_NAME.search(line.text.upper()) and re.fullmatch(r"[A-Za-z .]+", line.text.strip()):
            return line.text.strip()
    return None


def extract_dob(lines: list[TextLine]) -> str | None:
    """Date of birth: a date on or next to a DOB label, otherwise the earliest plausible birth date."""
    for i, line in enumerate(lines):
        if _DOB_LABEL.search(line.text):
            dates = find_dates(lines[i:i + 2])
            if dates:
                return dates[0].isoformat()
    today = date.today()
    births = sorted(d for d in find_dates(lines) if 1930 <= d.year <= today.year - 10)
    if births:
        return births[0].isoformat()
    m = _YEAR_OF_BIRTH.search(" ".join(line.text for line in lines))
    return m.group(1) if m else None


def extract_details(image: np.ndarray, first_name: str = "", last_name: str = "") -> dict:
    """What the candidate is asked to confirm: name, ID number and DOB read from the ID, plus the
    ID photo crop (BGR) when a face is found on the card."""
    portrait = find_portrait(image)
    lines = get_ocr_engine().read(image)
    id_type, id_number = extract_id_number(lines)
    return {
        "name": extract_name(lines, first_name, last_name),
        "id_number": id_number,
        "id_type": id_type,
        "dob": extract_dob(lines),
        "portrait_crop": crop_face(image, portrait, 0.35) if portrait else None,
        "text_lines": len(lines),
    }


# ------------------------------------------------------------------ main entry
def find_portrait(image: np.ndarray) -> Face | None:
    faces = get_face_engine().detect(image, max_side=1600, min_score=PORTRAIT_MIN_SCORE)
    return faces[0] if faces else None


def read_document(image: np.ndarray, *, first_name: str, last_name: str, profile_dob: date | None,
                  profile_embedding: np.ndarray | None) -> DocumentReading:
    quality = measure_quality(image)
    portrait = find_portrait(image)
    lines = get_ocr_engine().read(image)
    confident = [line for line in lines if line.confidence >= MIN_LINE_CONFIDENCE]
    chars = sum(len(re.sub(r"\W", "", line.text)) for line in confident)
    reading = DocumentReading(
        quality=quality,
        portrait=portrait,
        portrait_crop=crop_face(image, portrait, 0.35) if portrait else None,
        lines=lines,
        readable_lines=len(confident),
        text_readable=len(confident) >= MIN_READABLE_LINES and chars >= MIN_READABLE_CHARS,
    )
    if lines:
        reading.name = match_person_name(first_name, last_name, lines)
        dates = find_dates(lines)
        reading.dates_found = [d.isoformat() for d in dates]
        reading.dob_match = (profile_dob in dates) if dates and profile_dob else None
    if portrait is not None:
        engine = get_face_engine()
        reading.portrait_embedding = engine.embed(image, portrait)
        if profile_embedding is not None:
            reading.face_similarity_profile = engine.similarity(reading.portrait_embedding, profile_embedding)
    return reading
