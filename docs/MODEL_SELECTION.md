# TrustPRO model research and selection

Constraints for every model: CPU-only on a normal Windows laptop, Python 3.11, local inference
(no cloud API), pretrained weights available, maintained code, and a licence compatible with a
commercial product. Candidates were **measured on this project's own recordings** (real interview
recordings made with TrustPRO, the seven test videos in `Example_Videos/` - eye movements,
looking down, phones, earbuds, headphones, other people walking in - and Indian ID samples), not
chosen from published benchmarks alone.

Test machine: Intel Core i5-10210U (4 cores / 8 threads, 1.6 GHz), no GPU.

## Summary

| Capability | Selected | Licence | Measured cost |
|---|---|---|---|
| Gaze (live) | MediaPipe Face Landmarker (landmarks + eye blendshapes) + Intel `gaze-estimation-adas-0002` + `head-pose-estimation-adas-0001` (OpenVINO) | Apache-2.0 | 28-48 ms / frame |
| Face detection | YuNet 2023mar (OpenCV Zoo) | MIT | ~65 ms on a 960 px frame |
| Face recognition | ArcFace R50 / WebFace600K (InsightFace `buffalo_l`), switchable to SFace | ArcFace weights: **non-commercial**; SFace: Apache-2.0 | ~340 ms / face |
| ID text (OCR) | RapidOCR 3.9 with PP-OCRv5 mobile detection + English recognition | Apache-2.0 | 3-5 s / ID image |
| Objects: person, phone, laptop, screen, book | **RF-DETR Medium** for reports and **RF-DETR Small** for live checks (Roboflow, COCO), ONNX on OpenVINO; D-FINE Small as fallback | Apache-2.0 | Medium ~1 s / frame, ~1.2-1.4 frames/s with parallel streams; Small 0.73 s / frame on 2 threads |

## A. Gaze estimation

### Candidates

| Option | Notes | Verdict |
|---|---|---|
| **Intel Open Model Zoo `gaze-estimation-adas-0002`** | Learned 3-D gaze vector from two 60x60 eye crops + head pose. 1.9 M parameters, 0.14 GFLOPs, ~1 ms on CPU. Trained on Intel's own dataset (60 people), MAE 6.95 deg. Apache-2.0. | **Selected** |
| MediaPipe eye blendshapes (`eyeLook*`) | Used alone in the first version: noisy as a gaze direction. Now used as a *supporting* cue (eyes rolled down / turned within the head), where they are robust even when the eyelids hide the iris. | Supporting cue |
| Geometric iris-in-eye ratios | Classic approach; weak vertically because the eyelids move with the iris. | Not used |
| L2CS-Net / MobileGaze / UniGaze | Accurate, but weights trained on Gaze360 / MPIIFaceGaze, whose licences are research-only (CC BY-NC). | Rejected: licence |
| ETH-XGaze, OpenFace 2.0 | Non-commercial licences. | Rejected: licence |

### Pipeline (`app/ai/gaze.py`)

1. **MediaPipe Face Landmarker** (VIDEO mode) finds the faces, eye corners and eyelids, and gives
   blendshapes: `eyeLookDown` / `eyeLookIn` / `eyeLookOut` (where the eyes point *within the head*),
   `eyeBlink` and `mouthSmile`.
2. **Head pose**: Intel `head-pose-estimation-adas-0001` on the face crop, which is the same
   convention the gaze model was trained with (pitch + = head bent down).
3. **Gaze**: eye crops of 1.8 x eye width are de-rolled exactly as in Intel's reference demo and fed
   to `gaze-estimation-adas-0002`. The output is the *combined* head + eye direction, so a turned head
   with the eyes still on the screen is not reported as a look away. Its zero is the camera.
4. **Eyelid aperture** (lid opening / half eye width) from the landmarks.
5. **Calibration**: during the **10-second countdown** the candidate looks at the number in the
   middle of the screen. Robust two-pass medians of gaze, head pose, aperture, `eyeLookDown` and
   blink become that candidate's reference for "the screen". If no face is usable during the
   countdown, the first good seconds of the interview are used instead.
6. **Classification** relative to the reference, smoothed (EMA) with hysteresis; a direction is
   reported after it has held for **0.55 s** (four frames at 5 frames/s), which filters blinks:
   - **The camera is FORWARD.** A webcam sits on top of the screen, so the camera is `cam_dy =
     -reference pitch` degrees above the screen centre. UP needs a pitch change of at least
     `max(20, cam_dy + 12)` degrees - above the camera, not at it - or the head thrown back
     35 degrees or more.
   - **DOWN works with the iris hidden.** Looking below the screen lowers the eyelids over the
     iris, and the gaze model then reads *higher*, not lower (measured: -7 to -11 deg while the
     screen reference was -15.7). DOWN therefore uses three cues, any of which is enough, filtered
     over the last five frames (a one- or two-frame blink is ignored):
     - eyes rolled down: `eyeLookDown` up 0.30 from the countdown, with the lids lowering and the
       head not tilted back (`eyeLookDown` is relative to the head: tilting the head back while
       looking at the camera raises it too);
     - head bent down 15 degrees or more with the eyes down;
     - the gaze vector below the screen's lower edge (`max(cam_dy, 10) + 6` degrees below the
       reference).
     A broad smile squints the eyes; it is not counted as looking down.
   - **LEFT / RIGHT**: gaze yaw change >= 20 degrees, or >= 14 degrees when the eyes themselves are
     turned the same way (`eyeLookIn/Out`); with the eyes unreadable, a head turn of 30 degrees.
     Smiles widen the thresholds by 25%.
   - **The yaw reference follows the screen.** In a real test the head was turned 9 degrees during
     the countdown, which left a 15-degree offset and produced false LEFT looks for the whole
     interview. The reference now moves towards the median yaw of frames whose eyes look straight
     ahead within the head (sideways glances do not count) - at most 1 degree per second and at
     most 12 degrees from the countdown value.

### Evaluation (2026-10-05)

Five recordings were hand-labelled from large eye crops at 5 frames/s (1,012 frames): two real
TrustPRO interviews with their stored countdown calibration (interviews 30 and 32) and three
example videos (`Example_Videos/` 1, 3 and 5; calibrated on a stretch where the person looks at the
screen or camera). Two more TrustPRO interviews (25 and 29) were checked as timelines. Episode level
= the reported direction shows a labelled look away during it or within 1 s after.

| | Previous rules | Current rules |
|---|---|---|
| Per-frame agreement | 71.7% | **87.1%** |
| Looks down detected | 3 / 4 | **4 / 4** |
| Looks up detected | 3 / 4 | 3 / 4 |
| Looks left detected | 4 / 14 | **10 / 14** |
| Looks right detected | 3 / 11 | **10 / 11** |
| False alarms while looking at the screen or camera | 8.1% of frames | **1.1%** |

The previous rules reported **looking at the camera as UP** (74 frames) because the countdown
reference is the screen centre, about 15 degrees below the camera; they missed looking down
whenever the blink score crossed 0.6, because closed-looking eyes "kept the previous state". The
remaining misses are eye-only glances shorter than 0.6 s and one up-left look. The thresholds
were tuned on these recordings, so expect somewhat lower accuracy on new people.

**Limitations:**
- Thick glasses and strong side light reduce eye-crop quality (one example video with glasses
  reads +15 degrees for camera looks).
- A second screen right behind the webcam looks like forward.
- Glances shorter than 0.55 s are not reported.
- Looking only slightly above the camera (under ~10 degrees) stays FORWARD - deliberately, so that
  eye contact with the camera is never reported as UP.
- Calibration assumes the candidate looks at the countdown number.

## B. Face detection

**YuNet (OpenCV Zoo, MIT)** is built into OpenCV, gives the five landmarks needed for recognition
alignment, and found the portrait on every ID sample.

## C. Face recognition / identity matching

| Measured on local samples (2026-10-01) | SFace (Apache-2.0) | ArcFace R50 (`buffalo_l`) |
|---|---|---|
| Same person, recent ID card vs webcam | 0.909 | 0.936 |
| **Highest different-person score** | **0.437** | **0.205** |
| Threshold | 0.363 | 0.40 |

SFace scored two different men above its threshold, so **ArcFace is the default** (loaded directly
with ONNX Runtime, aligned with YuNet landmarks).

**Licence warning:** InsightFace's `buffalo_l` weights are for non-commercial research only.
Before commercial use, license them or set `FACE_ENGINE=sface`.

**ID photos need their own band.** Real PAN cards are small, printed and often years old. Genuine
pairs measured 0.25-0.35 on old PAN photos and 0.94-0.97 on a recent licence, while
different-person pairs were at most 0.21. ID-photo comparisons therefore report:
- **consistent** at 0.30 or above,
- **inconclusive (human review)** between 0.20 and 0.30,
- **not consistent** below 0.20.

Live face comparisons keep 0.40.

## D. ID text (OCR) and name matching

RapidOCR with PP-OCRv5 mobile detection and the English recogniser read the driving licence,
Aadhaar and PAN samples. Tesseract, EasyOCR, docTR, PaddleOCR (full framework) and vision-language
models were rejected for install weight, Windows friction or speed.

**Name matching** (`match_person_name`) is order-independent and tolerant:
- Every profile name part must be found in the same OCR window (up to 3 adjacent lines), using
  fuzzy matching per word.
- Initials match full names, so "B" matches "BHASKARA".
- Merged OCR words are handled, so "SRINIVASARAO" still matches.
- Extra words on the ID are allowed and reported.
- Real result: profile "Srinivas Boorada" matched the PAN text "BOORADA SRINIVAS" at 100%, with
  the order difference noted.
- The same surname appearing only on the father's-name line is not a match.

The date of birth matches if any date read from the ID equals the profile date. The parser
tolerates OCR gluing labels to dates, as in "/DO15/08/2002".

## E-H. Environment: people, phones, laptops, screens, printed material

| Model | Licence | Measured on this CPU | Phone results on test videos |
|---|---|---|---|
| **RF-DETR Medium** (COCO AP 54.7) | Apache-2.0 | 730 ms/frame; 1.2-1.4 frames/s in parallel | Real phones found. Headphone cups at most 0.48, earbuds at most 0.43. |
| D-FINE Small (COCO AP 48.5), previous default | Apache-2.0 | 340 ms/frame | Earbuds and headphone cups reported as phones at **0.71-0.78** (10 false frames in the earphone video) |
| YOLO11s (earlier prototype) | AGPL-3.0 | 243 ms/frame | Missed phones partly out of frame |

RF-DETR-M was exported from Roboflow's official `rfdetr` 1.11.1 package (`scripts/export_rfdetr.py`).
The ONNX post-processing was validated against `RFDETRMedium.predict` and agrees within 0.005. It
runs on **OpenVINO**:
- a LATENCY model limited to 2 threads for live checks, so gaze stays responsive,
- a THROUGHPUT model with parallel inference streams (`AsyncInferQueue`) for reports.

The model sometimes reads a phone held flat as "remote", so "remote" counts as a handheld device
at 0.55 or above.

**Relevance rules** (`app/pipeline/observations.py`), measured on the office recordings:

| Rule | Why |
|---|---|
| Phone long side >= 0.35 x candidate's face width | Earbuds were 0.14-0.21 and headphone cups about 0.57 at low confidence; a phone held near the face is about 0.7 or more |
| Additional person: box height >= 0.45 x the candidate's, or a second face >= half the candidate's | People at far desks measured 0.27-0.38; people right behind the candidate 0.47-0.96 |
| Laptop / screen box >= 6% of the frame | Laptops on desks behind the candidate covered 2.7-4.6% |
| Detection's own hand-inside-candidate box is not a person | The detector read a raised hand as a second person |

Single-frame detections must be confirmed on neighbouring frames (+/-0.4 s and 0.8 s). The report
lists how many detections each rule ignored.

**Result on the real office recordings:**
- In interview 18 the phone at the ear and the phone in hand were reported.
- 135 background-laptop detections were ignored.
- The ID card held up to the camera was reported as printed material.

**Limitations:** COCO has no earpiece class; a phone hidden in a hand is missed; posters of people
can count as persons.

## Performance (i5-10210U)

- **Live:** gaze takes 28-48 ms per frame. The environment check runs every 1.5 s on 2 threads.
  When the browser runs on the same laptop, live checks are slower than on a separate server.
- **Report:** a 117 s interview took 174 s and a 157 s interview took 177 s. Identity, environment
  and ID-document analysis run in parallel; RF-DETR-M dominates the time. Use
  `OBJECT_MODEL=dfine_s` for roughly 2x faster, less accurate environment analysis.
- **GPU later:** `ONNX_PROVIDERS` (ArcFace) and OpenVINO's `GPU` device can be enabled without code
  changes.
