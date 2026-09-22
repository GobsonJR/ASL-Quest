# ASL Quest — Phase 6 Report

Appearance-Decorrelation Experiment (Color/Brightness/Contrast/Saturation Augmentation)

Hypothesis tested: *"Does appearance-decorrelating augmentation reduce
signer-dependent representation and improve sign-discriminative
generalization?"*

**Bottom line: No. Test accuracy nudged up slightly (13.18% → 14.73%), but the
representation-level diagnosis is unchanged: class silhouette is still
negative and signer silhouette did not fall — it rose (0.205 → 0.2167). This
is Case B, not Case A: the small accuracy gain is not evidence of
sign-discriminative learning.**

---

## 1. Pre-flight

All checks below were run and passed before any training:

1. Inspected `src/asl_citizen/preprocessing.py` (spatial transforms + temporal
   sampling) and `src/asl_citizen/dataset.py` (how transforms are wired into
   `ASLCitizenDataset`/`build_dataloader`).
2. Inspected the Phase 4B configuration (`scripts/train_native_10.py`) and its
   recorded results (`outputs/asl_citizen_native_10/reports/`).
3. Inspected Phase 5 diagnostic artifacts
   (`outputs/asl_citizen_native_10/diagnostics/phase5_generalization/`),
   confirming the established silhouette numbers on disk:
   - frozen: class silhouette `-0.0725`, signer silhouette `0.205`
   - layer4: class silhouette `-0.0769`, signer silhouette `0.094`
4. Confirmed the only intended ML change is appearance augmentation — no
   architecture, LR, seed, split, temporal-sampling, or vocabulary change.
5. Verified the native_10 manifest is unchanged: `train.csv`=149 rows,
   `val.csv`=39 rows, `test.csv`=129 rows (byte-identical counts to Phase
   4B/4C).
6. Verified the official signer split: no train/val, train/test, or val/test
   participant overlap (re-checked at train time, matching Phase 4B/4C's own
   preflight assertions).
7. Verified the class mapping is unchanged (alphabetical):

   | Index | Gloss | Index | Gloss |
   |---:|---|---:|---|
   | 0 | BOOK | 5 | NO |
   | 1 | EAT1 | 6 | PLEASE |
   | 2 | HELLO | 7 | THANKYOU |
   | 3 | HELP | 8 | WATER |
   | 4 | MOTHER | 9 | YES |

8. Verified the output path is new and isolated:
   `outputs/asl_citizen_native_10/experiments/appearance_aug/` — distinct from
   Phase 4B (`outputs/asl_citizen_native_10/`), Phase 4C
   (`.../experiments/layer4/`), tiny-overfit checkpoints, and
   `outputs/asl_citizen_100/`. The training script refused to run if a best
   checkpoint already existed at that path (it did not).
9. Confirmed no A-Z, Word Spelling, frontend, backend API, database
   migration, or `asl_citizen_100` file was touched. `git`/`find` were not
   used to check a VCS diff (this project is not a git repo), so isolation
   was verified by construction: only three new files were created
   (`scripts/train_native_10_appearance_aug.py`,
   `scripts/evaluate_native_10_appearance_aug.py`,
   `scripts/diagnose_native_10_phase6.py`) and two shared files were edited
   additively (see §3).
10. Ran the existing ASL Citizen test suite: `tests/test_asl_citizen.py`, 18
    tests, **all pass**, both before and after the additive preprocessing/
    dataset edits described in §3.

Printed at the end of pre-flight: `SAFE TO START PHASE 6`.

## 2. Exact augmentation configuration

Printed by the training script immediately before training started:

```json
{
  "train_transform": {
    "temporal_sampling": "jittered_temporal_indices (unchanged, +/-2 frames)",
    "spatial_resize": [256, 256],
    "spatial_crop": "RandomResizedCrop, SAME crop for every frame in a clip (unchanged)",
    "random_resized_crop_scale": [0.85, 1.0],
    "random_resized_crop_ratio": [0.9, 1.1],
    "color_jitter": {
      "applied": true,
      "scope": "TRAIN ONLY; SAME sampled factors applied to every frame in a clip",
      "brightness": 0.2,
      "contrast": 0.2,
      "saturation": 0.2,
      "hue": 0.0,
      "horizontal_flip": false,
      "rotation": false,
      "perspective": false,
      "random_erasing": false,
      "blur": false,
      "noise": false
    },
    "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
  },
  "eval_transform": {
    "temporal_sampling": "uniform_temporal_indices (unchanged, deterministic)",
    "spatial_resize": [240, 240],
    "spatial_crop": "CenterCrop (unchanged, deterministic)",
    "color_jitter": {"applied": false, "note": "never applied to val/test"},
    "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
  }
}
```

Rationale for the magnitudes: `ColorJitter(brightness=0.2, contrast=0.2,
saturation=0.2)` — a conservative ±20% shift on each factor, no hue change
(hue can distort skin tone in ways unrelated to appearance decorrelation and
was explicitly out of scope). This is mild relative to the existing geometric
crop (which already removes at most ~15% of frame area), so hand/sign
visibility is preserved; only global appearance statistics shift.

Consistency: factors are sampled **once per clip** and applied identically to
all 16 frames (mirroring how the existing crop already applies one sampled
crop box to every frame in a clip), so no new frame-to-frame flicker is
introduced.

## 3. Changes made

Two shared files were edited **additively** (new optional parameters,
defaulting to `None`, preserving prior behavior exactly for every existing
caller/experiment):

- `src/asl_citizen/preprocessing.py`: added `apply_clip_color_jitter()` and an
  optional `color_jitter: ColorJitter | None = None` parameter to
  `apply_spatial_clip()`. When `None` (the default, used by every existing
  experiment), behavior is byte-identical to before this change.
- `src/asl_citizen/dataset.py`: added an optional `color_jitter` parameter to
  `ASLCitizenDataset.__init__` and `build_dataloader()`. The dataset forces
  `color_jitter = None` whenever `training=False`, so validation/test clips
  can never be jittered even if a caller passed a value in by mistake.

New files (Phase 6 only, no other file was modified):

- `scripts/train_native_10_appearance_aug.py` — isolated training run,
  modeled directly on `scripts/train_native_10_layer4.py`'s pattern.
- `scripts/evaluate_native_10_appearance_aug.py` — reuses
  `scripts/evaluate_native_10.py`'s unmodified `evaluate()`.
- `scripts/diagnose_native_10_phase6.py` — reuses
  `scripts/diagnose_native_10_phase5.py`'s unmodified
  `run_inference_with_features()`.

All new outputs are under
`outputs/asl_citizen_native_10/experiments/appearance_aug/`.

## 4. Tests

`tests/test_asl_citizen.py`, run via `python -m unittest tests.test_asl_citizen -v`:

- Before the preprocessing/dataset edits: **18/18 passed**.
- After the preprocessing/dataset edits: **18/18 passed** (confirms the
  additive change did not alter default behavior).
- A standalone smoke test (one train batch with `color_jitter` set, one val
  batch without it, forward + backward pass) completed without error before
  the full run was launched.

## 5. Training history

| | Phase 6 (appearance_aug) | Phase 4B (frozen, for reference) |
|---|---:|---:|
| Epochs completed | 23 (early stopped) | 23 (early stopped) |
| Best epoch | 11 | 11 |
| Stop reason | no `val_accuracy` improvement for 12 epochs | same |
| Train accuracy @ best epoch | 24.83% | 28.86% |
| Train loss @ best epoch | 1.9388 | 1.9508 |
| Train accuracy @ final epoch (23) | 50.34% | 58.39% |
| Train loss @ final epoch (23) | 1.3123 | 1.2263 |
| NaN/Inf detected | No | No |

Both runs stopped at the same epoch (23) with the same best epoch (11) and
the same early-stopping trigger. Phase 6's train accuracy is consistently a
few points lower than Phase 4B's at matching epochs — consistent with color
jitter acting as a mild regularizer that slows memorization of the tiny
training set, as expected from adding any augmentation.

## 6. Validation results

| | Phase 6 (appearance_aug) | Phase 4B (frozen) |
|---|---:|---:|
| Best validation accuracy | 17.95% (epoch 11) | 17.95% (epoch 11) |
| Validation top-5 @ best epoch | 48.72% | 48.72% |
| Validation loss @ best epoch | 2.4539 | 2.4083 |
| Unique predicted classes @ best epoch | 4 / 10 | 4 / 10 |

The best validation accuracy is numerically identical to Phase 4B
(17.95%, same epoch). This alone is a strong early signal that the
intervention did not meaningfully change what the model converged to.

## 7. Test results

| Metric | Phase 6 (appearance_aug) | Phase 4B (frozen) | Phase 4C (layer4) |
|---|---:|---:|---:|
| Top-1 accuracy | **14.73%** | 13.18% | 9.30% |
| Top-5 accuracy | 48.84% | 51.16% | 48.06% |
| Macro precision | 0.1165 | 0.0942 | 0.0530 |
| Macro recall | 0.1601 | 0.1446 | 0.0999 |
| Macro F1 | **0.1045** | 0.0866 | 0.0653 |
| Unique predicted classes | 7 / 10 | 7 / 10 | 7 / 10 |
| Dominant predicted class | YES | YES | NO |
| Dominant class share | **0.6434** | 0.6434 | 0.3178 |

Per-class precision/recall/F1 (Phase 6):

| Gloss | Support | Precision | Recall | F1 | Predicted count |
|---|---:|---:|---:|---:|---:|
| BOOK | 12 | 0.2353 | 0.3333 | 0.2759 | 17 |
| EAT1 | 17 | 0.1667 | 0.1176 | 0.1379 | 12 |
| HELLO | 14 | 0.0000 | 0.0000 | 0.0000 | 2 |
| HELP | 13 | 0.0000 | 0.0000 | 0.0000 | 1 |
| MOTHER | 15 | 0.5000 | 0.2667 | 0.3478 | 8 |
| NO | 11 | 0.0000 | 0.0000 | 0.0000 | 0 |
| PLEASE | 13 | 0.0000 | 0.0000 | 0.0000 | 0 |
| THANKYOU | 12 | 0.1667 | 0.0833 | 0.1111 | 6 |
| WATER | 12 | 0.0000 | 0.0000 | 0.0000 | 0 |
| YES | 10 | 0.0964 | 0.8000 | 0.1720 | 83 |

`NO`, `PLEASE`, and `WATER` are **never predicted at all** — the identical
set of three classes that Phase 4B never predicted at all. `YES` alone
absorbs 83 of 129 test predictions (64%), effectively unchanged from Phase
4B's dominant-class share (also 0.6434, to four decimal places).

## 8. Confusion matrix

Rows = true class, columns = predicted class, order
`[BOOK, EAT1, HELLO, HELP, MOTHER, NO, PLEASE, THANKYOU, WATER, YES]`:

```
BOOK      [4, 1, 0, 0, 0, 0, 0, 1, 0, 6]
EAT1      [2, 2, 0, 0, 1, 0, 0, 0, 0, 12]
HELLO     [2, 0, 0, 0, 1, 0, 0, 0, 0, 11]
HELP      [1, 1, 0, 0, 0, 0, 0, 0, 0, 11]
MOTHER    [0, 2, 0, 0, 4, 0, 0, 1, 0, 8]
NO        [1, 2, 0, 1, 0, 0, 0, 1, 0, 6]
PLEASE    [2, 1, 1, 0, 0, 0, 0, 1, 0, 8]
THANKYOU  [3, 1, 1, 0, 1, 0, 0, 1, 0, 5]
WATER     [1, 2, 0, 0, 0, 0, 0, 1, 0, 8]
YES       [1, 0, 0, 0, 1, 0, 0, 0, 0, 8]
```

`YES` is the majority prediction for **every single true class**, including
its own row — the collapse pattern from Phase 4B/4C is intact.

## 9. Prediction distribution

```
BOOK: 17   EAT1: 12   HELLO: 2   HELP: 1   MOTHER: 8
NO: 0      PLEASE: 0  THANKYOU: 6   WATER: 0   YES: 83
```

7 of 10 classes are used at all; 3 are never predicted — identical to Phase
4B's set of never-predicted classes (`NO`, `PLEASE`, `WATER`).

## 10. Per-signer analysis

| Signer | n | Accuracy (Phase 6) | Unique preds (Phase 6) | Accuracy (Phase 4B) | Unique preds (Phase 4B) |
|---|---:|---:|---:|---:|---:|
| P15 | 11 | 0.18 | 3 | 0.27 | 3 |
| P17 | 10 | 0.20 | 3 | 0.20 | 3 |
| P18 | 11 | 0.09 | 3 | 0.09 | 4 |
| P22 | 10 | 0.10 | 3 | 0.20 | 2 |
| P35 | 14 | 0.14 | 2 | 0.07 | 3 |
| P42 | 13 | 0.08 | 3 | 0.08 | 2 |
| P47 | 11 | 0.09 | 2 | 0.18 | 2 |
| P48 | 12 | 0.08 | 3 | 0.17 | 4 |
| P49 | 10 | **0.10** | **1** | **0.10** | **1** |
| P6  | 13 | 0.15 | 5 | 0.08 | 2 |
| P9  | 14 | 0.36 | 3 | 0.07 | 3 |

Signer P49 still collapses to a single predicted class (`YES`, 10/10) in
*both* Phase 6 and Phase 4B — an exact match. Most other signers still
concentrate 2–3 predicted classes dominated by `YES`. A few individual
signers shifted a little (P9 improved from 7% to 36%, P22 dropped from 20%
to 10%), but there is no systematic reduction in per-signer collapse across
the group — signer accuracy min/max for Phase 6 is 0.0769–0.3571, a similar
spread to Phase 4B.

## 11. Class silhouette analysis

| | Phase 4B (frozen) | Phase 4C (layer4) | Phase 6 (appearance_aug) |
|---|---:|---:|---:|
| Silhouette score by true class | -0.0725 | -0.0769 | **-0.0723** |

Class silhouette is essentially unchanged (Δ = +0.0002, statistically
negligible for a 129-sample, 512-dim feature space). Classes remain
**not separable** in the learned feature space — clusters overlap so heavily
that the mean sample is closer, on average, to samples of other classes than
to its own class centroid neighbors.

## 12. Signer silhouette analysis

| | Phase 4B (frozen) | Phase 4C (layer4) | Phase 6 (appearance_aug) |
|---|---:|---:|---:|
| Silhouette score by signer | 0.205 | 0.094 | **0.2167** |

Signer silhouette **increased** slightly relative to Phase 4B (0.205 →
0.2167), rather than decreasing. Appearance augmentation did not weaken the
signer-clustered structure in the learned representation — if anything, the
representation is marginally *more* signer-clustered than the un-augmented
frozen baseline.

## 13. PCA comparison

Saved to
`outputs/asl_citizen_native_10/experiments/appearance_aug/diagnostics/`:

- `appearance_aug_pca_by_class.png` — 2D PCA projection of test features,
  colored by true class. Visually the same diffuse, non-separated cloud seen
  in the Phase 5 frozen/layer4 plots — no new class-aligned clustering
  structure emerged.
- `appearance_aug_pca_by_signer.png` — same projection colored by signer.
  Visible signer-aligned sub-clusters persist, consistent with the
  silhouette numbers above.

(Comparable Phase 4B/4C plots are at
`outputs/asl_citizen_native_10/diagnostics/phase5_generalization/feature_analysis/`.)

## 14. Phase 4B vs Phase 4C vs Phase 6

| Metric | Phase 4B Frozen | Phase 4C Layer4 | Phase 6 Appearance Aug |
|---|---:|---:|---:|
| Best validation accuracy | 17.95% | 17.95% | 17.95% |
| Test top-1 accuracy | 13.18% | 9.30% | 14.73% |
| Test top-5 accuracy | 51.16% | 48.06% | 48.84% |
| Macro F1 | 0.0866 | 0.0653 | 0.1045 |
| Train accuracy (final epoch) | 58.39% | — | 50.34% |
| Unique test classes predicted | 7 / 10 | 7 / 10 | 7 / 10 |
| Dominant predicted class | YES | NO | YES |
| Dominant class share | 0.6434 | 0.3178 | 0.6434 |
| Class silhouette | -0.0725 | -0.0769 | -0.0723 |
| Signer silhouette | 0.205 | 0.094 | 0.2167 |

**No model here should be called "best" on accuracy alone.** Phase 6 has the
highest test top-1 and macro F1 of the three, but:

- Best validation accuracy is *identical* across all three runs (17.95%,
  same epoch for 4B/6) — the small test-accuracy differences are within the
  noise band of a 129-sample test set (each single correct/incorrect
  prediction shifts top-1 by ~0.78 percentage points).
- Phase 6's representation is not more class-separated (class silhouette
  essentially flat) and not less signer-clustered (signer silhouette rose,
  not fell) relative to Phase 4B.
- Phase 4C's lower accuracy is explained by a different, already-diagnosed
  cause (layer4 fine-tuning on 149 samples overfits/destabilizes faster than
  a frozen backbone — see the Phase 5 report), not by anything this
  experiment tests.
- Phase 6's dominant class and its exact share (0.6434) match Phase 4B to
  four decimal places, and the never-predicted class set (`NO`, `PLEASE`,
  `WATER`) is identical — strong evidence the two runs converged to
  essentially the same solution.

## 15. Diagnosis

Appearance augmentation produced a small, likely noise-level improvement in
test accuracy and macro F1, with no corresponding change in the
representation-level diagnosis from Phase 5:

- Class silhouette remains strongly negative and did not improve
  meaningfully (-0.0725 → -0.0723).
- Signer silhouette did not decrease — it increased slightly
  (0.205 → 0.2167).
- Per-signer prediction collapse is essentially unchanged (P49 still
  predicts a single class 10/10; most other signers still concentrate on
  2–3 classes dominated by `YES`).
- The confusion matrix, dominant class, dominant class share, and
  never-predicted class set are all nearly or exactly identical to Phase 4B.

Notably, Phase 6's top-5 accuracy (48.84%) is actually *lower* than Phase
4B's (51.16%) even though top-1 is higher — the two metrics move in opposite
directions, which is another sign that the top-1 delta is more consistent
with sampling noise on a 129-clip test set than with a systematic
improvement in the learned representation.

This matches **CASE B** from the experiment's own interpretation framework:
*accuracy improved, but signer silhouette remains high and class silhouette
remains poor — the improvement may not indicate genuine sign-discriminative
learning.* The most likely explanation is that mild color jitter acted as a
generic regularizer (train accuracy at every epoch is a few points lower
than Phase 4B's, consistent with less overfitting to the training set) that
produced a small, not-necessarily-meaningful test-set fluctuation, rather
than a change in what visual cues the model relies on.

**Restating the dataset limitation, as instructed:** this result does not
show that appearance augmentation "fixed" anything. Native-10 training data
still contains only ~1.0–1.3 clips per signer per class. With so few signers
per class, brightness/contrast/saturation jitter changes the *appearance* of
a clip but does not create new *signer diversity* — the model can still
learn "signer X's overall look" as a shortcut for class Y, because it never
sees class Y performed by more than a couple of distinct people regardless
of how each individual clip's colors are perturbed. The Phase 5 diagnostic
already flagged signers-per-class (not raw clip count, not appearance
variance within a clip) as the binding constraint, and this experiment's
outcome is consistent with that: an augmentation that does not add
signer diversity did not measurably reduce the signer/class confound.

## 16. Evidence for/against appearance-shortcut reduction

**Against (dominant conclusion):**
- Signer silhouette increased, not decreased.
- Class silhouette is statistically unchanged.
- Per-signer prediction collapse patterns are essentially identical
  (including an exact match on which signer collapses to 1 class and which
  three classes are never predicted at all).
- Dominant class and its share match Phase 4B almost exactly.

**For (weak, does not outweigh the above):**
- Test top-1 accuracy and macro F1 both increased modestly.
- Train accuracy was lower at every epoch than Phase 4B, consistent with a
  genuine (if generic) regularization effect from the augmentation, not an
  implementation bug.
- No NaN/Inf, no crash, no collapse to a single class in validation
  diagnostics during training.

Net assessment: appearance augmentation alone, at the tested moderate
strength, is **not** meaningfully addressing the signer/class confound
identified in Phase 5. The dataset's signers-per-class limitation remains
the primary bottleneck.

## 17. Recommended next experiment

*(Not launched — reported only, per the Phase 6 stop condition.)*

Given the Phase 5 diagnosis explicitly attributed the confound to
signers-per-class rather than to raw clip count or within-clip appearance
variance, the next diagnostic step (not a training run) would be to quantify
whether the small Phase 6 accuracy delta survives a stricter significance
check (e.g., bootstrap confidence intervals over the 129-sample test set,
or a repeated run with a different seed to see whether 14.73% vs 13.18% is
distinguishable from run-to-run noise at this dataset size) before drawing
any conclusion — including a negative one — with high confidence.

---

**STOP.** Per the Phase 6 protocol, no further training (another
augmentation strength, layer4, full fine-tuning, 20-class, 50-class, or
production integration) is launched automatically.
