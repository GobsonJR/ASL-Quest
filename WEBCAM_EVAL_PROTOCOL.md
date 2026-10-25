# Held-out webcam evaluation: protocol and collection checklist

Purpose: decide, on evidence, whether a MediaPipe-landmark recognizer is comparable to or better than the production ResNet18 on real
webcam use. **ResNet18 stays the default until this evaluation is COMPLETE and you approve a change.** Nothing here trains, replaces or promotes a model.

Status today: **not complete.** The data on disk is 94 eligible frames, 9 letters, no recorded person, one room, one lighting condition, no
negatives. `scripts\evaluate_webcam_heldout.py coverage` prints exactly what is missing. Numbers from that pilot are not a verdict.

## What is compared

Three systems, scored on the identical frame list (each frame is detected once; every system answers from that one detection):

| system | path |
|---|---|
| ResNet18 (production) | MediaPipe hand box -> padded square crop -> ResNet18 (`models/best_model.pth`) |
| Landmark L1 | MediaPipe 21 landmarks -> 87 geometric features -> sklearn (`models/experiments/landmark_L1_s0.joblib`) |
| Landmark L2 | same, `landmark_L2_s0.joblib` |

`resnet_full` (ResNet18 on the whole frame, no MediaPipe) is reported only as a reference.

## What counts as held-out

A frame is eligible only if its **session and person** were not used to train any model:

* ResNet18: trained on the Kaggle dataset only (the run refuses if the checkpoint declares webcam training).
* Landmark models: their own metadata lists the training rooms (`original-block*` = Kaggle train split, plus webcam room `plain-wall` for L2). Those sessions are excluded automatically.
* People: every session records a **person**. A person who appears in any training session is excluded. The old `plain-wall` session has no person recorded, so
  whoever signed it must be passed as `--training-person NAME` (or simply not be one of the evaluation people). The report lists this limitation explicitly.

Selection of thresholds or models on this data is not allowed: it is a test set. (The app's 0.70 confidence threshold is fixed; a sweep is reported for information.)

## Required coverage (the report says COMPLETE only if all are met)

| requirement | why |
|---|---|
| every letter A-Z >= 30 labelled frames, J and Z included | per-letter recall at n=30 still has a ~ +-15 point interval |
| >= 3 named people, each >= 100 frames | person-to-person spread is the main generalisation risk |
| >= 3 rooms | background generalisation |
| >= 3 lighting conditions, each >= 100 frames | lighting |
| near, medium and far, each >= 100 frames | distance |
| left and right hand each >= 200 frames; every letter >= 5 left-hand frames | the left-hand failure seen on the dataset |
| >= 100 no-hand frames and >= 100 partial-hand frames | detection recall / false detections / behaviour at the frame edge |
| >= 50 labelled frames with the hand touching the frame edge | hands partly out of frame |
| each horizontal and vertical third of the frame >= 100 detected frames | hand position (derived from landmarks after detection) |

J and Z are recorded as still shapes (the app does not support motion). Their results measure still-pose recognition only.

## Metrics reported (per system, identical frames)

* hand-detection recall on labelled frames; detection rate on no-hand (a false detection) and partial-hand frames
* end-to-end accuracy (a missed hand counts as wrong) and accuracy conditional on detection, with Wilson 95% intervals and paired exact tests vs ResNet18
* per-letter recall and precision (precision = correct among answers of that letter)
* confidence-threshold errors at the app's 0.70: wrong-and-accepted, correct-but-rejected, accepted answers on frames with no hand; and a 0.5-0.9 sweep
* breakdowns by person, room, lighting, background, distance, hand, glasses, edge contact, measured brightness, hand position and hand size
* latency: MediaPipe, crop, ResNet18 (GPU and CPU), landmark classifier

Measured and reproducible: the report records the git commit, SHA-256 of every model file, and a digest over the SHA-256 of every evaluated frame.

## Collection plan (about 940 labelled frames + 200 negatives, roughly 3 x 1 hour)

Three **new** people (not whoever signed `plain-wall`), each recording **three sessions**, each session in a different room (`room`) and a different
lighting condition (`lighting`). Use new room names (not `plain-wall`). Per session, per letter A-Z, save **4 frames**:

1. right hand, **near**, hand in the **left third** of the frame
2. right hand, **medium**, hand in the **centre**
3. right hand, **far**, hand in the **right third**
4. **left** hand, any distance (vary it between letters)

Vary hand height (top / middle / bottom third) from letter to letter. Per session also:

* for **8 different letters**, one frame with the hand clearly **cut off by the frame edge** (the collector warns "hand touches the frame edge" and still saves it)
* **12 NO-HAND frames** (both hands out of view: empty scene, your face, torso, an object in hand, hands in lap, ...). Change the scene each time: near-identical frames are refused
* **12 PARTIAL-HAND frames** (a hand mostly outside the frame, fingertips or wrist entering at an edge)

Totals: 9 sessions x (104 labelled + 24 negatives) = 936 labelled, 216 negatives. Use "Hold" captures **not at all** (they are not needed here).

Rules that keep the test honest:

* The person who signs a frame is the person named in the session. Do not mix people inside a session.
* Confirm a letter only if the hand really shows it. The label is yours; the recognizer is never consulted.
* Do not look at, tune against, or retrain on these sessions. Do not delete or edit them (the evaluation never does).
* If you are unsure of a sign, skip the frame.

## Step 0 - camera mirror check (once, 2 minutes)

The investigation found that MediaPipe's raw "Left/Right" label matched the hand the person stated in 257 of 259 real frames, the opposite of what the code assumes.
Whether your camera delivers mirrored frames decides if that is a camera effect or a MediaPipe convention. Create a session with room `mirror-check`,
person `mirror-check`, hold a sheet with a large word (e.g. `ASL RIGHT`) beside your hand, save any one frame, then open the saved JPEG
(`dataset\webcam\<session id>\<letter>\<sample>.jpg`) and note whether the text reads normally or mirrored. Tell me the answer. Add
`--exclude-session <that session id>` to every evaluation command so this session is not scored.

## Exact commands

Run one heavy job at a time on this machine (the backend and an evaluation together exhaust memory).

Collect (PowerShell, from the project root):

```powershell
$env:ASL_QUEST_DEV_COLLECT = "1"
venv311\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --env-file .env
```

second terminal:

```powershell
cd frontend
npm run dev
```

Open http://localhost:5173 and go to **More -> Collect data**. For each session: fill **Room**, **Lighting**, **Background**, **Person signing**, set *Face visible* and *Glasses* to match reality, press **Start session**. Choose a letter, set **Hand** and **Distance**, press **Capture**, check the crop and verdict, press **Yes, this is X**.
Use **Save NO-HAND frame** and **Save PARTIAL-HAND frame** for the negatives (no letter needed). Stop the backend (Ctrl+C) when finished.

Check progress at any time (light, no model loaded):

```powershell
venv311\Scripts\python.exe scripts\evaluate_webcam_heldout.py coverage --training-person <name of whoever signed plain-wall> --exclude-session <mirror-check session id>
```

When it says `COVERAGE MET`, evaluate (heavy: loads torch and MediaPipe; the backend must be stopped):

```powershell
venv311\Scripts\python.exe scripts\evaluate_webcam_heldout.py run --require-complete --training-person <name> --exclude-session <mirror-check session id>
venv311\Scripts\python.exe scripts\evaluate_webcam_heldout.py report --training-person <name> --exclude-session <mirror-check session id>
```

Results: `outputs\webcam_heldout_eval\REPORT.md` (tables) and `manifest.json` (frame hashes, model hashes, git commit).

## What this evaluation will not tell you

* Different cameras, phones, or tablets (record `camera` in the session if you add them).
* Moving J and Z (the app does not support motion).
* Anything beyond the 3 people and 3 rooms collected; more diverse data can change the picture.
