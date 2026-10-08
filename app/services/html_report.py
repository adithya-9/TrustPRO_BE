"""Self-contained HTML report (images embedded), written next to the assessment recording:
storage/candidates/<candidate id>/assessments/<assessment id>/report_<report id>.html
"""
from __future__ import annotations

import base64
from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np

from app.services.storage import encode_jpeg

RESULT_TEXT = {"CONSISTENT": ("Match", "ok"), "NOT_CONSISTENT": ("No match", "bad"), "MIXED": ("Partly matching", "warn"),
               "INCONCLUSIVE": ("Unclear", "warn"), "UNAVAILABLE": ("Not available", "na")}


def _img(data: bytes | np.ndarray | None, alt: str) -> str:
    if data is None:
        return f'<div class="noimg">{escape(alt)}: not available</div>'
    if isinstance(data, np.ndarray):
        data = encode_jpeg(data, quality=85, max_side=720)
    return f'<img alt="{escape(alt)}" src="data:image/jpeg;base64,{base64.b64encode(data).decode()}">'


def _clock(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _badge(ok: bool | None, yes: str, no: str, unknown: str) -> str:
    if ok is None:
        return f'<span class="badge na">{escape(unknown)}</span>'
    return f'<span class="badge {"ok" if ok else "bad"}">{escape(yes if ok else no)}</span>'


def _result(result: str | None) -> str:
    """Only the outcome badge; similarity scores are kept in the database, not shown."""
    label, cls = RESULT_TEXT.get(result or "UNAVAILABLE", (result or "-", "na"))
    return f'<span class="badge {cls}">{escape(label)}</span>'


def write(path: Path, *, inputs: dict, identity: dict, environment: dict, environment_events: list,
          identity_evidence: list, report_id: int) -> Path:
    cand = inputs["candidate"]
    name = " ".join(x for x in (cand.get("first_name"), cand.get("last_name")) if x) or "Candidate"
    details = inputs.get("id_details") or {}
    doc = identity.get("id_document") or {}
    checks = doc.get("checks") or {}
    comparisons = {c["pair"]: c for c in identity.get("comparisons", [])}
    best_frame = next((e.jpeg for e in identity_evidence if e.label == "HIGHEST_SIMILARITY"), None)
    started = inputs.get("started_at")

    # ---------------------------------------------------------------- identity
    rows = [
        ("Name on the ID", details.get("name")),
        ("ID number", details.get("id_number")),
        ("ID type", details.get("id_type")),
        ("Date of birth on the ID", details.get("dob")),
    ]
    details_html = "".join(f"<tr><th>{escape(k)}</th><td>{escape(str(v)) if v else '<span class=muted>not read</span>'}</td></tr>"
                           for k, v in rows)
    profile_rows = [
        ("Name", name),
        ("Date of birth", cand.get("date_of_birth").isoformat() if cand.get("date_of_birth") else None),
        ("Mobile", cand.get("mobile_number")),
        ("City", cand.get("city")),
    ]
    profile_html = "".join(f"<tr><th>{escape(k)}</th><td>{escape(str(v)) if v else '-'}</td></tr>" for k, v in profile_rows)

    name_check = checks.get("name") or {}
    face_id = checks.get("face_vs_profile") or {}
    pv, iv = comparisons.get("PROFILE_VIDEO", {}), comparisons.get("ID_VIDEO", {})

    checks_html = f"""
      <tr><th>Name on the ID matches the profile</th><td>{_badge(name_check.get('matched') if name_check else None, 'Match', 'No match', 'Could not read')}</td></tr>
      <tr><th>Date of birth on the ID matches the profile</th><td>{_badge(checks.get('dob_match'), 'Match', 'No match', 'Could not read')}</td></tr>
      <tr><th>ID photo vs profile photo</th><td>{_result(face_id.get('result'))}</td></tr>
      <tr><th>Profile photo vs assessment video</th><td>{_result(pv.get('result'))}</td></tr>
      <tr><th>ID photo vs assessment video</th><td>{_result(iv.get('result'))}</td></tr>"""
    if identity.get("status") == "failed":
        checks_html += f'<tr><th>Face comparison</th><td><span class="badge bad">{escape(identity.get("error", "failed"))}</span></td></tr>'

    # ---------------------------------------------------------------- environment
    if environment.get("status") == "failed":
        env_html = f'<p class="badge bad">{escape(environment.get("error", "Environment analysis failed"))}</p>'
    elif not environment_events:
        env_html = (f'<p class="clean">No environmental anomalies were detected '
                    f'({environment.get("frames_analyzed", 0)} frames analysed).</p>')
    else:
        counts = "".join(f'<span class="chip">{escape(k)}: {v}</span>' for k, v in environment.get("events_by_type", {}).items())
        cards = "".join(
            f'<figure>{_img(e.jpeg, e.type)}<figcaption><b>{escape(e.type)}</b><span>{_clock(e.timestamp_s)}'
            f'{"" if e.confidence is None else f" · {e.confidence * 100:.0f}%"}</span></figcaption></figure>'
            for e in environment_events)
        env_html = f'<div class="chips">{counts}</div><div class="grid">{cards}</div>'

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Assessment report - {escape(name)}</title>
<style>
  body {{ font-family: Inter, Segoe UI, system-ui, sans-serif; background: #f6f7fb; color: #172033; margin: 0; }}
  main {{ max-width: 1100px; margin: 0 auto; padding: 28px 16px 60px; }}
  h1 {{ font-size: 24px; margin: 0 0 4px; }} h2 {{ font-size: 18px; margin: 0 0 14px; }}
  .muted {{ color: #6b7280; font-size: 13px; }}
  section {{ background: #fff; border-radius: 16px; padding: 22px; margin-top: 20px; box-shadow: 0 1px 3px rgba(0,0,0,.06); }}
  .photos {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-bottom: 18px; }}
  .photos figure, .grid figure {{ margin: 0; background: #f1f3f8; border-radius: 12px; overflow: hidden; }}
  .photos img {{ width: 100%; height: 220px; object-fit: contain; background: #111; display: block; }}
  .photos figcaption {{ padding: 8px 10px; font-size: 13px; font-weight: 600; }}
  .cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  th {{ text-align: left; font-weight: 500; color: #4b5563; padding: 9px 8px; border-bottom: 1px solid #eef0f4; width: 55%; }}
  td {{ padding: 9px 8px; border-bottom: 1px solid #eef0f4; font-weight: 600; }}
  .badge {{ display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 13px; font-weight: 600; }}
  .ok {{ background: #dcfce7; color: #166534; }} .bad {{ background: #fee2e2; color: #991b1b; }}
  .warn {{ background: #fef3c7; color: #92400e; }} .na {{ background: #eef0f4; color: #4b5563; }}
  .chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 14px; }}
  .chip {{ background: #fff1f2; color: #9f1239; border-radius: 999px; padding: 4px 12px; font-size: 13px; font-weight: 600; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 14px; }}
  .grid img {{ width: 100%; display: block; }}
  .grid figcaption {{ padding: 8px 10px; font-size: 13px; display: flex; justify-content: space-between; gap: 8px; }}
  .clean {{ background: #dcfce7; color: #166534; padding: 14px; border-radius: 12px; font-weight: 600; }}
  .noimg {{ height: 220px; display: grid; place-items: center; color: #6b7280; font-size: 13px; }}
  @media (max-width: 760px) {{ .photos, .cols {{ grid-template-columns: 1fr; }} }}
</style></head>
<body><main>
  <h1>Assessment report - {escape(name)}</h1>
  <p class="muted">Assessment {inputs['interview_id']} · candidate {inputs['candidate_id']} ·
     {escape(started.strftime('%d %b %Y %H:%M') + ' UTC') if started else ''} · duration {_clock(inputs['duration_ms'] / 1000)} ·
     report {report_id} generated {datetime.now().strftime('%d %b %Y %H:%M')}</p>

  <section>
    <h2>1. ID verification</h2>
    <div class="photos">
      <figure>{_img(inputs.get('profile_image'), 'Profile photo')}<figcaption>Profile photo</figcaption></figure>
      <figure>{_img(inputs.get('portrait_image') if inputs.get('portrait_image') is not None else inputs.get('capture_image'), 'ID photo')}<figcaption>Photo on the ID</figcaption></figure>
      <figure>{_img(best_frame, 'Assessment video')}<figcaption>Assessment video</figcaption></figure>
    </div>
    <div class="cols">
      <div><table><caption class="muted" style="text-align:left;padding-bottom:6px">Read from the ID and saved by the candidate</caption>{details_html}</table>
           <table style="margin-top:14px"><caption class="muted" style="text-align:left;padding-bottom:6px">Profile</caption>{profile_html}</table></div>
      <div><table><caption class="muted" style="text-align:left;padding-bottom:6px">Checks</caption>{checks_html}</table></div>
    </div>
  </section>

  <section>
    <h2>2. Environmental anomalies</h2>
    {env_html}
  </section>
</main></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path
