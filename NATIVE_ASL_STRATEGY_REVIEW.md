# Native ASL Recognition Strategy Review

Architecture/data planning only. No training was launched, no dataset,
manifest, model, backend, frontend, or database file was changed while
producing this report.

---

## 1. Why current training failed

Three controlled experiments (Phase 4B frozen, Phase 4C layer4, Phase 6
appearance augmentation) all trained the **same** architecture — a
frozen/partially-unfrozen **ImageNet-pretrained** ResNet18 CNN feeding a
GRU, trained from scratch on the GRU/classifier head — on the **same** 149
training clips across 10 classes. All three converged to the same failure
mode: a ~4-out-of-10-class collapse dominated by one or two classes
(`YES` at 64% of test predictions in 4B and Phase 6; `NO` at 32% in 4C),
never predicting 3 classes at all (`NO`, `PLEASE`, `WATER`), and a feature
space with **negative class silhouette** (classes are not separable) and
**positive, non-decreasing signer silhouette** (signers *are* separable).

A fourth, previously-run local experiment adds decisive evidence that this
is not specific to the 10-class subset: `outputs/asl_citizen_100/` trained
the identical architecture on the full 100-class subset and scored **0.83%
top-1 / 5.19% top-5 — statistically indistinguishable from the 1%/5% random
baseline** (`outputs/asl_citizen_100/diagnostics/phase7d_diagnosis_report.json`).
On a 200-sample probe, the best checkpoint predicted only 2 unique classes
across the entire probe, with 197/200 predictions going to a single class
(`BOY`). Scaling the vocabulary up with the same architecture did not
dilute the signer shortcut — it produced total collapse instead, because
more classes with the same 52-signer pool only shrinks
clips-per-signer-per-class further.

**Root cause, stated plainly:** an ImageNet-pretrained 2D CNN has never
learned to distinguish sign-relevant motion from person-specific appearance
(skin tone, clothing, framing, background, build). With ~1 training clip
per signer per class, gradient descent has no reason to prefer the
sign-motion explanation over the (equally predictive, on this data) signer
identity explanation. This is an architecture/pretraining-domain-mismatch
problem compounded by a data-scale problem, not a hyperparameter problem —
which is why three hyperparameter-and-augmentation experiments produced the
same failure pattern three times.

## 2. Evidence for signer shortcut

Consolidated from Phase 5 and Phase 6 (all read from existing diagnostic
JSON, nothing re-run for this report):

| Signal | Phase 4B | Phase 4C | Phase 6 |
|---|---:|---:|---:|
| Class silhouette (test features) | -0.0725 | -0.0769 | -0.0723 |
| Signer silhouette (test features) | 0.205 | 0.094 | 0.2167 |
| Dominant predicted class / share | YES / 64.34% | NO / 31.78% | YES / 64.34% |
| Never-predicted classes | NO, PLEASE, WATER | BOOK, HELLO, PLEASE | NO, PLEASE, WATER |
| Signer P49 (10 test clips) | 100% predicted as YES | — | 100% predicted as YES |

A representation that separates people better than it separates signs,
across three independent architectural/augmentation variants, with two of
them producing an *exactly* matching failure signature (same dominant
class, same share to 4 decimal places, same never-predicted set) — that
consistency is itself evidence the model is converging on the same
signer-keyed solution regardless of which of these three levers is pulled.

## 3. ASL Citizen data limitations

Quantified directly from the manifest CSVs and the full ASL Citizen source
corpus (`load_all_source_rows()`), not estimated:

**Native-10 split, per split:**

| Split | Clips | Unique signers | Clips/class | Signers/class |
|---|---:|---:|---:|---:|
| Train | 149 | 26 | 13–18 (mean 14.9) | 12–16 (mean 14.2) |
| Val | 39 | 5 | 3–5 | 3–4 |
| Test | 129 | 11 | 10–17 | 10–11 (near-total coverage) |

- Train mean clips-per-signer-per-class: **1.0–1.3** across all 10 classes
  (exact values in
  `outputs/asl_citizen_native_10/diagnostics/phase5_generalization/dataset_size_report.json`).
- **42 of the corpus's 52 total signers** appear somewhere in the native-10
  subset (26 in train, 5 in val, 11 in test, no overlap).
- Test-set signers see almost every class once (10 or 11 of the 11 test
  signers per class) — the opposite regime from train, which is exactly why
  a signer-keyed shortcut that works on train collapses on test: the test
  signers were never seen with their correct class label during training.

**Is native-10's vocabulary unusually poor coverage? No.** Checked against
the full corpus (2,731 glosses, 83,399 videos, 52 signers, mean 30.5
videos/gloss, median 31):

| Gloss | Videos (full corpus) | Percentile vs. all 2,731 glosses |
|---|---:|---:|
| BOOK | 31 | 46th |
| EAT1 | 39 | 99th |
| HELLO | 32 | 91st |
| HELP | 31 | 46th |
| MOTHER | 32 | 91st |
| NO | 30 | 9th |
| PLEASE | 31 | 46th |
| THANKYOU | 31 | 46th |
| WATER | 30 | 9th |
| YES | 30 | 9th |

Every native-10 gloss sits at or above the corpus median except NO/WATER/YES
(9th percentile — still not pathological; 9th percentile of 2,731 glosses
still has 30 videos, near the corpus mean). **The 10 classes are ordinary,
representative ASL Citizen glosses, not a cherry-picked hard subset.**

**The deeper finding:** checking unique signers per gloss across the *whole*
corpus (not just native-10's split) gives 26–30 signers per native-10 gloss
out of 52 total, and a corpus-wide mean of **28 unique signers per gloss**.
With ~30 videos and ~28 signers per typical gloss, ASL Citizen is
structurally a **single-take elicitation dataset** — each signer performs
each gloss roughly once, across the *entire* 2,731-gloss corpus, not just
our subset. This means the "~1 clip/signer/class" ceiling is not an
artifact of how we sliced 10 classes out of the corpus; it is how the
dataset was collected. No amount of re-partitioning the existing native-10
signers/clips into train/val/test will raise clips-per-signer-per-class
above ~1, because the source data doesn't contain repeated same-signer
takes of the same gloss to draw on.

## 4. Model initialization options

See §8 for the full candidate table. Headline conclusion: ASL Citizen's own
released baselines (I3D and ST-GCN, both MIT-licensed, both fine-tuned on
this exact 52-signer dataset) already demonstrate that a **63.1% top-1**
accuracy is achievable on the *full* 2,731-way signer-independent task with
this data — using a backbone that was pretrained on video action
recognition (Kinetics) or on pose sequences, not on static ImageNet frames.
That is the single strongest piece of evidence in this review: **the
signer/class confound is not unsolvable on ASL Citizen** — it was solved by
a different architecture on the same signers, same clips, same
one-take-per-signer structure we just quantified in §3. Our from-scratch
frozen-ImageNet-CNN + GRU pipeline is the thing that has been unable to
learn appearance-invariant sign features from this data, not the data
itself.

## 5. Dataset strategy options

**A. ASL Citizen only (current approach), unchanged architecture.**
Already falsified three times (Phase 4B/4C/6) plus the 100-class scale-up
(§1). Not viable without an architecture/pretraining change.

**B. ASL Citizen + pretrained sign/video representation.**
Directly addresses the diagnosed root cause (§1): swap the from-scratch
ImageNet-CNN+GRU encoder for a backbone that was pretrained on video motion
(Kinetics) or ASL Citizen itself, then fine-tune only the classifier
(or lightly fine-tune the backbone) on the native-10 subset. Requires no
new data collection, reuses the existing signer-independent split, and is
testable in a single, cheap experiment (§9–10). This is the option
supported by the strongest direct evidence (§4).

**C. Controlled multi-signer dataset collected specifically for native-10.**
Directly increases signers-per-class and repeats-per-signer, attacking the
data-scale side of the problem instead of the architecture side. Section 7
gives concrete numbers. This does not require touching ASL Citizen and is
compatible with either architecture — but it is the slower, higher-effort
option and, per §4, may not even be necessary if B resolves the confound.

**Recommendation ordering:** B first, because it is evidence-backed,
cheap, fast, and directly tests the diagnosed root cause without any new
data-collection effort. C is the fallback if B's fine-tuned representation
*still* shows the signer/class confound (i.e., B alone isn't sufficient,
meaning the data-scale problem is binding even for a good backbone) — see
§9 for how the two combine.

## 6. Recommended evaluation protocol

The existing native-10 official split is **already signer-independent** by
construction (verified in every phase's pre-flight: zero train/val,
train/test, val/test participant overlap) and should be kept for any B
experiment, so results stay comparable to Phase 4B/4C/6.

For any new Option-C data collection, the same rule applies at collection
time, not just at split time:

```
TRAIN       : signers {S1 ... Sk}         — never seen in VAL or TEST
VALIDATION  : signers {Sk+1 ... Sk+m}     — never seen in TRAIN or TEST
TEST        : signers {Sk+m+1 ... Sk+m+n} — never seen in TRAIN or VAL
```

Concretely: assign every recruited signer to exactly one split *before*
recording, not after — otherwise it is tempting to backfill "whichever
split needs more clips" from a signer already used elsewhere, which
reintroduces leakage. Log signer ID directly in the manifest at collection
time (already the convention this project uses).

Metrics to report for every future native-10 experiment (matches what
Phase 4B/4C/6 already report, kept as the fixed standard):

- Top-1 accuracy, top-5 accuracy
- Macro precision, macro recall, macro F1 (per-class, not just aggregate)
- Full confusion matrix
- Per-signer accuracy and per-signer predicted-class distribution
  (this is the collapse detector — a model can have plausible aggregate
  accuracy while still being signer-keyed, as Phase 6 just showed)
- Prediction distribution + dominant-class share (collapse detector #2)
- Class silhouette and signer silhouette on extracted features (collapse
  detector #3, the only one of the three that inspects the representation
  directly rather than its output)

No single metric should be trusted alone — Phase 6 is the concrete example
of accuracy moving one direction while the representation-level signal
moved the other way.

## 7. Practical dataset requirements

Grounded in three established signer-independent ISLR benchmarks, not
invented:

| Dataset | Signs | Signers | Reps/signer/sign | Total clips |
|---|---:|---:|---:|---:|
| LSA64 | 64 | 10 | 5 | 3,200 |
| AUTSL | 226 | 43 | ~3.9 (avg) | 38,336 |
| Native-10 (current) | 10 | 26 (train) | ~1.0–1.3 | 149 (train) |

Native-10 sits far below both precedents on repeats-per-signer, and below
AUTSL (though not LSA64) on signer count. Because native-10 only has 10
classes — far fewer than either benchmark — matching their per-class
density is a small, tractable collection effort, not a large one:

| Tier | Signers | Reps/signer/sign | Clips/class | Total (10 classes) | Rationale |
|---|---:|---:|---:|---:|---:|
| **Minimum viable** | 10 | 3 | 30 | 300 | Matches LSA64's signer count and lands between LSA64/AUTSL on repeats; roughly 2x native-10's current clips/class, but with actual repeat structure instead of near-singleton coverage |
| **Recommended** | 20 | 4 | 80 | 800 | Matches AUTSL's per-sign signer diversity and repeat rate; large enough that no single signer's appearance can dominate a class's gradient signal |
| **Strong** | 35 | 5 | 175 | 1,750 | Exceeds AUTSL's per-sign density; would support held-out signer counts (§6) of 20 train / 8 val / 7 test without starving any split |

All tiers assume the collection protocol enforces splits at recruitment
time (§6), not after the fact. These numbers describe raw *collection*
effort only — they say nothing about attainable accuracy, which is
correctly left unestimated (framework Task 3 explicitly forbids inventing
accuracy claims).

## 8. Candidate pretrained models

| Model | Pretrained domain | Input format | Temporal support | GPU memory | License | Integration complexity | Fit for isolated ASL signs | Runs on RTX 4050 6GB | Fine-tunable on our data |
|---|---|---|---|---|---|---|---|---|---|
| **Microsoft ASL Citizen I3D baseline** (`microsoft/ASL-citizen-code`) | ASL Citizen itself (Kinetics-pretrained I3D, then fine-tuned on the same 52-signer, 2,731-gloss corpus we already use) | RGB clips | Native (3D conv, full clip) | ~5.1 GB reported for I3D-class models | **MIT** | Low–medium — same dataset/domain/signer split as our pipeline; would replace the encoder, reuse our existing dataloader's video decoding | **Best fit** — same domain, same signers, demonstrated 63.1% top-1 on the full 2,731-way signer-independent task | Yes (fits alongside batch 2–4, 16 frames) | Yes — head/last-block fine-tune on 149 clips, or feature extraction + lightweight classifier |
| **Microsoft ASL Citizen ST-GCN baseline** (same repo) | ASL Citizen, skeleton/pose sequences | Pose/hand keypoint sequences (needs a landmark extractor — MediaPipe already installed locally) | Native (graph-temporal) | <1 GB typical for ST-GCN-scale models | **MIT** | Medium — needs a pose-extraction preprocessing step we don't currently run for native-10 (though MediaPipe is already used for the A-Z hand-crop path) | Very strong — appearance-invariant by construction, directly targets the diagnosed signer shortcut | Yes, trivially | Yes |
| **WLASL Kinetics→WLASL I3D** (`dxli94/WLASL`) | Kinetics-400 → WLASL-2000 (2,000 ASL signs, different signer pool) | RGB clips | Native (3D conv) | ~5.1 GB | Research/academic use only (not MIT — check repo terms before any production use) | Medium — different preprocessing conventions, would need adaptation | Good — ASL-specific, but signer pool doesn't overlap ours (still a useful pretrained *initialization*, just not a same-domain checkpoint like the ASL Citizen one) | Yes | Yes, as a second-choice initialization if the ASL Citizen checkpoint is unavailable |
| **OpenHands (AI4Bharat) pose-pretrained ST-GCN/SL-GCN** | Self-supervised pretraining on Indian-SL pose sequences, with demonstrated cross-lingual transfer | Pose/hand keypoint sequences | Native (graph-temporal) | <1 GB | Apache/MIT-style (AI4Bharat convention) — verify on checkout | Medium-high — repo is archived/unmaintained, may need patching for current PyTorch | Good — same appearance-invariance argument as the ASL-Citizen ST-GCN option, plus published cross-lingual transfer evidence | Yes, trivially | Yes, in principle — higher integration risk due to unmaintained status |
| **torchvision.models.video** (`r3d_18`, `mc3_18`, `r2plus1d_18`, `s3d`, `mvit_v2_s`) | Kinetics-400, general action recognition (not sign-specific) | RGB clips | Native (3D conv or video transformer) | S3D/R3D: modest; MViT: larger | Torchvision's standard permissive license (BSD-3) | **Lowest** — already installed, no new dependency, weights fetch through the existing torch hub cache | Moderate — generic action-recognition features, not sign-specific, but still a *video* motion prior instead of ImageNet's *static-image* prior | Yes for R3D/S3D/MC3; MViT is tighter but workable at small batch | Yes |
| **Self-supervised video transformers for ISLR** (VideoMAE-style / SignBERT+) | Self-supervised video pretraining, some sign-specific variants published | RGB clips (ViT patches) | Native (transformer) | Base-size ViT video models are the heaviest option here | Mixed — check per-checkpoint | High — newest, least mature integration path of this list | Good on paper, unproven for our pipeline | Tight — feasible only with small batch/frame-subsampling, most at-risk of OOM on 6GB | Possible but higher risk; not a first choice |

Notes:
- No download was performed for this review; the Microsoft ASL Citizen
  repository is confirmed to host v1 checkpoints on its GitHub Releases
  page, MIT-licensed, but the repo itself is now archived/read-only
  (archived June 2026) — the code still works, but no further updates
  should be expected, and the checkpoint should be mirrored locally once
  downloaded rather than depended on remotely long-term.
- MediaPipe is **already installed in this project's venv** (`mediapipe
  1.0.1`) and already used for the A-Z hand-crop pipeline
  (`src/inference/hands.py`), so any pose/skeleton-based option (ST-GCN,
  SL-GCN) requires zero new heavyweight dependencies — only a new
  landmark-extraction step for native-10 clips, which is a data
  preprocessing change, not a production-pipeline change.

## 9. Recommended next experiment

**Do not launch. Recommended for a future phase only.**

**Fine-tune the Microsoft ASL Citizen I3D baseline checkpoint (§8, row 1)
on the existing native-10 signer-independent split, using the exact
evaluation protocol from §6, and compare directly against Phase 4B/4C/6.**

Why this one, over the alternatives:

- It is the only candidate that is simultaneously (a) pretrained on video
  *motion*, not static ImageNet frames, (b) pretrained on ASL specifically,
  and (c) pretrained on the *same 52 signers and the same one-take-per-signer
  data regime* we just quantified as the binding constraint in §3 — and it
  still reached 63.1% top-1 on a 2,731-way task under that constraint. That
  is direct, dataset-matched proof the confound is beatable with a better
  encoder, not a claim by analogy from an unrelated dataset.
- It requires **no new data collection** (§5, Option B), so it is testable
  immediately once approved, with results comparable to three already-run
  baselines.
- It is the cheapest failure-mode check: if a same-domain, already-proven
  encoder *still* shows the signer-shortcut signature (negative class
  silhouette, high signer silhouette, YES/NO-style collapse) on native-10
  specifically, that would be strong evidence the bottleneck really is
  data scale (§3's ~1 clip/signer/class ceiling) and not architecture —
  which would justify moving to Option C (§7) with confidence instead of
  guessing.

If this experiment is approved and run, and it *still* shows the confound,
the natural Phase 8 candidate becomes a **minimum-viable Option C
collection (§7)** paired with the same pretrained I3D encoder — i.e., use
the pretrained backbone to reduce how much new data is needed, rather than
choosing between B and C as mutually exclusive.

The ST-GCN/skeleton route (§8, row 2) is a strong secondary candidate and
arguably lower-risk (appearance-invariant by construction, tiny compute
footprint), but was not chosen as the *first* recommended experiment
because it requires building a new pose-extraction step for native-10 clips
that does not exist yet, whereas the I3D route reuses the existing
video-clip dataloader almost unchanged. It should be the immediate follow-up
if the I3D experiment is ambiguous.

## 10. Expected compute/time requirements

Not launched — estimated from this project's own recorded run times for
comparably-sized experiments (Phase 4B/4C/6, all ~60–80 sec/epoch on this
RTX 4050 for a 149-clip train set at batch size 4):

- **I3D fine-tune (recommended experiment, §9):** Similar per-epoch cost
  order of magnitude to Phase 4B/4C/6 (same 149 train clips, same 16-frame
  clips, same batch size 4) — a frozen-backbone/head-only fine-tune would
  likely be *faster* per epoch than our current ResNet18+GRU (fewer
  sequential per-frame CNN passes if the I3D backbone processes the whole
  clip as one 3D-conv forward pass rather than 16 independent 2D passes).
  Expect a comparable or smaller wall-clock budget to Phase 6's full run
  (23 epochs to early-stopping, ~30 minutes total).
- **ST-GCN/skeleton route (§9 follow-up):** Substantially cheaper — pose
  extraction is a one-time preprocessing pass over 149+39+129 = 317 clips
  (MediaPipe landmark extraction is CPU-feasible and fast per clip), and
  the resulting classifier is tiny (low-dimensional keypoint sequences,
  not 224×224 RGB frames), so training epochs would likely run in seconds
  rather than the ~70 sec/epoch seen for the RGB pipeline.
- **Option C data collection (§7), if pursued in parallel:** Not a compute
  cost but a scheduling one — the "recommended" tier (20 signers × 4 reps
  × 10 signs = 800 clips) is a recruitment and recording-session planning
  effort, independent of and running in parallel with any model work above.

---

**STOP.** Per this phase's scope, this document is architecture/data
planning only. No native-10 training was run, the manifest and signer
split were not touched, no model file was changed, and A-Z, Word Spelling,
the frontend, backend APIs, the database, and the chatbot were not
modified.
