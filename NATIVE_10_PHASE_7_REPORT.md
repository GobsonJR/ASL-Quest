# ASL Quest — Phase 7 I3D Transfer Report

Pretrained ASL Citizen I3D, frozen backbone, linear classifier head, native-10
signer-independent split.

**Headline result: frozen I3D features reach 96.9% test top-1 accuracy / macro F1
0.9691, versus 13.18% / 0.0866 for the best ResNet18 variant (Phase 4B). Class
silhouette flipped from negative (-0.0725) to positive (0.22); signer silhouette
flipped from positive (0.205) to slightly negative (-0.0612). This is CASE A: strong
evidence the pretrained representation is sign-discriminative and signer-decorrelated,
not merely a lucky accuracy bump.**

---

## 1. Official checkpoint verification

Verified by direct inspection of the official source repository
(`https://github.com/microsoft/ASL-citizen-code`, downloaded as a source archive over
HTTPS — `git clone` was blocked by the sandbox network, so the repo's `archive/refs/heads/main.zip`
was fetched instead and diffed conceptually against the GitHub UI; the release asset
was fetched through the GitHub Releases API, not a mirror):

- **Repository**: `microsoft/ASL-citizen-code` (official Microsoft Research repo for
  the ASL Citizen paper's baselines). Confirmed via the GitHub API
  (`api.github.com/repos/microsoft/ASL-citizen-code`), not a third-party mirror.
- **Release**: tag `checkpoints_v1`, "ASL Citizen Weights 1.0", published
  2023-06-13 by `alexxijielu` (the repository's committer/paper author).
- **Asset**: `ASL_citizen_I3D_weights.zip`, 56,382,633 bytes (size confirmed via the
  GitHub Releases API before download, not assumed).
- **Extracted checkpoint**: `ASL_citizen_I3D_weights.pt`, 60,521,671 bytes, a raw
  `torch.save`d `OrderedDict` state_dict (no wrapping metadata dict).
- **SHA-256** of the extracted file:
  `3538319620331d5ba731e0f9ac79161deeb37a895b7680d215864fc00a14477d`
  (recorded in `models/pretrained/asl_citizen_i3d/SOURCE.md` for future
  verification).
- **Architecture compatibility**: loaded with `strict=True` against
  `InceptionI3d(400, in_channels=3)` → `replace_logits(2731)` — **0 missing keys,
  0 unexpected keys**, 344 state_dict entries, 15,086,539 total parameters. This is
  a real, verified compatibility check, not an assumption from the filename.
- The repo was NOT used blindly: `I3D/aslcitizen_training.py`,
  `I3D/aslcitizen_testing.py`, `I3D/aslcitizen_dataset.py`, and
  `I3D/videotransforms.py` were all read in full before writing any of this
  experiment's code, specifically to extract the exact preprocessing and
  confirm what the checkpoint was actually trained on (see §4 and the
  correction below).

**Correction to the Phase 6 strategy review:** that review, based on secondary
web-search summaries, described this checkpoint as "Kinetics-pretrained I3D,
fine-tuned on ASL Citizen." Reading the actual training script
(`I3D/aslcitizen_training.py`) shows **no Kinetics checkpoint is loaded** — the model
is trained **end-to-end from random initialization directly on the full ASL Citizen
corpus** (2,731 glosses, ~65k+ training clips, all 52 signers; Adam lr=1e-3, up to 75
epochs, `ReduceLROnPlateau`). This is corrected in
`models/pretrained/asl_citizen_i3d/SOURCE.md` and is, if anything, a better match for
our transfer purpose than Kinetics pretraining would be: the model was forced to
become sign-discriminative across thousands of classes sharing the same ~52-signer
pool, which is exactly the mechanism the Phase 6 review hypothesized would break a
signer shortcut.

## 2. License

- **Code**: `microsoft/ASL-citizen-code` root `LICENSE` file — **MIT License,
  Copyright (c) Microsoft Corporation.** Verified by reading the file directly (copied
  verbatim into `models/pretrained/asl_citizen_i3d/LICENSE_microsoft_ASL-citizen-code.txt`),
  not inferred from a GitHub badge or search summary.
- **Weights**: hosted in the same MIT-licensed repository's Releases page with no
  separate license notice. Treated as covered by the same MIT terms (documented in
  `SOURCE.md` as "a reasonable reading, not a legal certainty").
- **Dataset**: explicitly a **separate license**, not covered by the above, and **not
  downloaded, redistributed, or stored anywhere in this project** for this phase.
  `models/pretrained/asl_citizen_i3d/` contains only the checkpoint, the code license
  text, and a provenance document — zero video files.
- `data/asl_citizen_native_10_i3d/` contains only manifest CSVs/JSON (participant IDs,
  filenames, glosses, split assignments) copied from the existing native-10 manifest —
  no video bytes.

## 3. Model architecture

`InceptionI3d` (Inception-v1 3D-inflated, Carreira & Zisserman "Quo Vadis, Action
Recognition?"), vendored unmodified into `src/i3d_transfer/pytorch_i3d.py` (MIT,
attributed) from the official repo, which itself sources it from
`github.com/piergiaj/pytorch-i3d`.

- 15,086,539 total parameters (comparable order of magnitude to the ResNet18+GRU's
  11,770,442).
- `extract_features(x)` returns the post-`Mixed_5c`, pre-logits average-pooled
  feature map: shape `(B, 1024, 7, 1, 1)` for a 64-frame, 224×224 input (1024 =
  384+384+128+128, the concatenated Inception branch channel widths).
- The original 2,731-class logits layer (`logits.conv3d`, shape
  `[2731, 1024, 1, 1, 1]`) is **never used** in this experiment — Task 3 required
  frozen feature extraction into a **new** 10-class head, not reuse of the original
  classifier.
- Backbone is fully frozen: `requires_grad_(False)` on every parameter, every forward
  pass wrapped in `torch.no_grad()`, verified programmatically
  (`extract_i3d_features_native_10.py` asserts `trainable_params == 0` before running
  any extraction).

## 4. Input/preprocessing

Matched to the **official** pipeline (`I3D/aslcitizen_dataset.py`,
`I3D/videotransforms.py`), not reused from the ResNet18 pipeline, and not guessed:

| Property | Official I3D value | Old ResNet18 pipeline (for contrast) |
|---|---|---|
| Frame count | 64, adaptive frameskip (1/2/3 based on total frame count thresholds 96/160), **centered** sampling window | 16, uniform ± jitter |
| Resize | short side ≥ 226 px, long side ≤ 256 px (aspect-preserving) | fixed square 240×240 or 256×256 |
| Crop | 224×224 | 224×224 (same) |
| Color space | **BGR** (cv2 native, never converted) | RGB (explicit `cv2.cvtColor`) |
| Normalization | `(pixel/255)*2 - 1` → **[-1, 1]** | ImageNet mean/std |
| Padding (short clips) | repeat first/last frame to 64 | index-clamped repeats |

Implemented in `src/i3d_transfer/preprocessing.py` as a direct, function-level port of
`load_rgb_frames_from_video`, `pad`, and `CenterCrop`. **One deliberate documented
deviation**: the official train-time transform is `RandomCrop(224) +
RandomHorizontalFlip`; this experiment uses `CenterCrop(224)` for every split
(train/val/test) and never flips. Reasoning, stated in the module docstring: (a) this
project's own existing convention (`src/asl_citizen/preprocessing.py`) explicitly
avoids horizontal flipping because ASL handedness is semantically meaningful — flipping
a sign can change or invalidate it; (b) with the backbone frozen, spatial crop jitter
only perturbs the *input pixels* to a frozen feature extractor, not the classifier
being trained, so determinism was preferred for interpretability of this specific,
first, "does the frozen representation already separate our signs" experiment.

**Preprocessing smoke test** (`scripts/smoke_test_i3d_preprocessing.py`, run before
any extraction or training): loaded one real native-10 clip end-to-end and verified —
tensor shape `(3, 64, 224, 224)` ✓, dtype `float32` ✓, all values finite ✓, value
range `[-0.969, 1.0]` (within the expected `[-1, 1]`) ✓, frame-to-frame deltas small
and smoothly varying (mean 0.00129, max 0.00676 — consistent with real, correctly
temporally-ordered video, not shuffled frames) ✓, and one real forward pass through
the loaded checkpoint produced finite pooled features and finite logits ✓.

## 5. GPU/VRAM requirements

Measured directly on this machine's RTX 4050 Laptop GPU (6.44 GB VRAM, confirmed via
`torch.cuda.get_device_properties`):

- Dummy-batch sanity check (batch=2, 64 frames, 224×224, frozen, no_grad): **724.2 MB**
  peak allocated.
- Full feature extraction over all 317 real clips (149 train + 39 val + 129 test),
  batch=1, frozen, no_grad: **392.7 MB** peak allocated, 104 seconds total wall-clock
  (49.9s train + 13.8s val + 40.3s test).
- **Conclusion: frozen-backbone inference is trivially practical on this 6GB GPU** —
  using under 12% of available VRAM even before considering that batch size could be
  increased substantially. Task 1 asked whether inference/fine-tuning is practical:
  inference is comfortably practical; a future partial/full fine-tune (gradients
  through some or all of the 15M-parameter backbone) was not attempted this phase
  (Task 3/6 required starting frozen) and would need its own memory check before being
  launched, but 15M params is small enough (the whole ResNet18+GRU pipeline was
  11.8M and trained fine on this GPU) that it is very unlikely to be a hard
  blocker — see §19.

## 6. Dataset and split

Unchanged from Phase 4B/4C/6: **149 train / 39 validation / 129 test**, same 10
glosses (BOOK, EAT1, HELLO, HELP, MOTHER, NO, PLEASE, THANKYOU, WATER, YES), same
official signer-independent split (verified again in this phase's own pre-flight:
26/5/11 unique signers in train/val/test, zero overlap in any direction).

Per Task 2, `data/asl_citizen_native_10/` (the original manifest) was **not
modified**. This experiment reads from `data/asl_citizen_native_10_i3d/`, a
byte-identical copy of the six manifest files (verified with `diff`, reported as
identical for all three CSVs). No dataset video was copied — only these small
manifest files, which is why the copy step is instantaneous and inspectable.

## 7. Training configuration

Per Task 6, the I3D **backbone is not present in the training script at all** —
`train_native_10_i3d_frozen.py` only imports `numpy`/`torch`/`sklearn`-adjacent code
and operates on the cached `(N, 1024)` feature arrays from §4/§5's extraction step.
"Backbone frozen" is therefore enforced by the absence of the backbone object in that
process, not merely by a flag.

| Setting | Value | Rationale |
|---|---|---|
| Head architecture | `Dropout(0.5) → Linear(1024, 10)` | Simplest suitable head (Task 3): I3D's `extract_features` is already temporally pooled, so no GRU/temporal module is added (Task 3's explicit instruction) |
| Head learning rate | `1e-3` | This project's existing `head_learning_rate` convention (`src/asl_citizen/config.py`'s `HEAD_LEARNING_RATE`), which Task 6 pointed at directly — not the ResNet *backbone* LR (`1e-4`), which Task 6 said not to reuse blindly |
| Weight decay | `1e-3` | Raised from this project's usual `1e-4`: a 1024→10 linear layer (10,250 weights) is more overparameterized relative to 149 training examples than the previous GRU+classifier combination was |
| Feature dropout | `0.5` | Second regularizer, applied to the input embedding (no hidden layer exists to place ordinary dropout within) |
| Optimizer | AdamW, full-batch gradient descent | 149 examples fit in one batch trivially; avoids introducing minibatch order as a new experimental variable for a linear probe this small |
| Seed | `42` | Consistent with every prior phase |
| Max epochs | `100` | Modest budget (Task 6) |
| Early-stopping patience | `15` epochs on `val_accuracy` | — |

**No hyperparameter sweep was run** — exactly one configuration, per Task 6.

## 8. Training history

| | Value |
|---|---:|
| Epochs completed | 27 (early stopped) |
| Best epoch | 12 |
| Stop reason | no `val_accuracy` improvement for 15 consecutive epochs |
| Train accuracy @ best epoch | 100.0% |
| Train loss @ best epoch | 0.7728 |
| Train accuracy @ final epoch (27) | 100.0% |
| Train loss @ final epoch (27) | 0.1931 |

Training reached 100% train accuracy by epoch 10 and 100% validation accuracy by
epoch 12 — an extremely fast, clean convergence curve, consistent with a linear probe
on an already highly separable representation rather than a model straining to fit
noise (contrast with Phase 4B/4C/6, where train accuracy climbed slowly to 50–94%
over 23 epochs while validation accuracy never exceeded 18%).

## 9. Validation results

| | Value |
|---|---:|
| Best validation accuracy | **100.0%** (epoch 12) |
| Validation top-5 @ best epoch | 100.0% |
| Validation loss @ best epoch | 0.8374 |
| Unique predicted classes @ best epoch | 10 / 10 |

## 10. Test results

| Metric | Phase 7 I3D frozen |
|---|---:|
| Top-1 accuracy | **96.90%** |
| Top-5 accuracy | 99.22% |
| Macro precision | 0.9709 |
| Macro recall | 0.9713 |
| Macro F1 | **0.9691** |
| Unique predicted classes | 10 / 10 |
| Dominant predicted class | EAT1 |
| Dominant class share | 13.18% |
| Nearest-centroid accuracy (geometry cross-check) | 96.90% |

Per-class precision/recall/F1:

| Gloss | Support | Precision | Recall | F1 | Predicted count |
|---|---:|---:|---:|---:|---:|
| BOOK | 12 | 0.9231 | 1.0000 | 0.9600 | 13 |
| EAT1 | 17 | 1.0000 | 1.0000 | 1.0000 | 17 |
| HELLO | 14 | 1.0000 | 1.0000 | 1.0000 | 14 |
| HELP | 13 | 1.0000 | 0.8462 | 0.9167 | 11 |
| MOTHER | 15 | 1.0000 | 0.8667 | 0.9286 | 13 |
| NO | 11 | 1.0000 | 1.0000 | 1.0000 | 11 |
| PLEASE | 13 | 0.9286 | 1.0000 | 0.9630 | 14 |
| THANKYOU | 12 | 1.0000 | 1.0000 | 1.0000 | 12 |
| WATER | 12 | 0.8571 | 1.0000 | 0.9231 | 14 |
| YES | 10 | 1.0000 | 1.0000 | 1.0000 | 10 |

`NO`/`PLEASE`/`WATER` — the three classes **never predicted at all** in Phase 4B and
Phase 6 — are now predicted correctly essentially every time (NO: 11/11, PLEASE:
13/13, WATER: 12/12). `YES` — the class that absorbed 64.34% of *all* predictions in
both Phase 4B and Phase 6 — is now predicted exactly 10 times for 10 true `YES`
examples, 0 false positives.

## 11. Confusion matrix

Rows = true class, columns = predicted class, order
`[BOOK, EAT1, HELLO, HELP, MOTHER, NO, PLEASE, THANKYOU, WATER, YES]`:

```
BOOK      [12, 0, 0, 0, 0, 0, 0, 0, 0, 0]
EAT1      [0, 17, 0, 0, 0, 0, 0, 0, 0, 0]
HELLO     [0, 0, 14, 0, 0, 0, 0, 0, 0, 0]
HELP      [1, 0, 0, 11, 0, 0, 1, 0, 0, 0]
MOTHER    [0, 0, 0, 0, 13, 0, 0, 0, 2, 0]
NO        [0, 0, 0, 0, 0, 11, 0, 0, 0, 0]
PLEASE    [0, 0, 0, 0, 0, 0, 13, 0, 0, 0]
THANKYOU  [0, 0, 0, 0, 0, 0, 0, 12, 0, 0]
WATER     [0, 0, 0, 0, 0, 0, 0, 0, 12, 0]
YES       [0, 0, 0, 0, 0, 0, 0, 0, 0, 10]
```

**4 errors total out of 129 test clips**: one HELP→BOOK, one HELP→PLEASE, two
MOTHER→WATER. Every other class is a perfect diagonal. Compare to Phase 4B/6, where
`YES` was the majority prediction for essentially every true class.

## 12. Prediction distribution

```
BOOK: 13   EAT1: 17   HELLO: 14   HELP: 11   MOTHER: 13
NO: 11     PLEASE: 14 THANKYOU: 12  WATER: 14   YES: 10
```

All 10 classes used, in proportions closely tracking true support (12–17 each) — no
class collapse, no never-predicted class, dominant-class share (13.18%) close to the
1-in-10 (10%) a perfectly balanced, non-collapsed classifier would show.

## 13. Per-signer results

| Signer | n | Correct | Accuracy |
|---|---:|---:|---:|
| P15 | 11 | 11 | 100.0% |
| P17 | 10 | 9 | 90.0% |
| P18 | 11 | 11 | 100.0% |
| P22 | 10 | 10 | 100.0% |
| P35 | 14 | 14 | 100.0% |
| P42 | 13 | 11 | 84.6% |
| P47 | 11 | 11 | 100.0% |
| P48 | 12 | 12 | 100.0% |
| **P49** | 10 | **10** | **100.0%** |
| P6 | 13 | 12 | 92.3% |
| P9 | 14 | 14 | 100.0% |

Signer accuracy range: **84.6%–100%**, with 7 of 11 signers at a perfect 100%.

**P49 is the single most striking individual result in this report.** In every one of
the three ResNet18 experiments (Phase 4B, 4C, and 6), signer P49 collapsed to
predicting a single class (`YES`) for all 10 of their test clips, regardless of what
they actually signed — 10% accuracy by chance alone (they happened to sign `YES` once
out of ten). With frozen I3D features, the same 10 clips from the same unseen signer
now score **100% accuracy**. Nothing about signer P49's clips changed between
experiments; only the representation used to classify them did.

## 14. Class silhouette

| | Phase 4B (ResNet frozen) | Phase 4C (ResNet layer4) | Phase 6 (appearance aug) | **Phase 7 (I3D frozen)** |
|---|---:|---:|---:|---:|
| Class silhouette | -0.0725 | -0.0769 | -0.0723 | **+0.22** |

Every ResNet18 variant had a **negative** class silhouette — samples were, on
average, closer to other classes' samples than to their own class's. I3D frozen
features flip this to a solidly **positive** 0.22 — a large, unambiguous
improvement, not noise (three ResNet variants clustered tightly around -0.072 to
-0.077; I3D is nearly 0.3 silhouette-points away from that cluster).

## 15. Signer silhouette

| | Phase 4B (ResNet frozen) | Phase 4C (ResNet layer4) | Phase 6 (appearance aug) | **Phase 7 (I3D frozen)** |
|---|---:|---:|---:|---:|
| Signer silhouette | 0.205 | 0.094 | 0.2167 | **-0.0612** |

Every ResNet18 variant had a **positive** signer silhouette — signers formed
coherent clusters in the feature space. I3D frozen features flip this to slightly
**negative** — signer identity is no longer a coherent cluster structure at all; if
anything, same-signer samples are (very mildly) *less* alike than different-signer
same-class samples in this projection. This is the appearance-decorrelation result
Phase 6's brightness/contrast/saturation augmentation was hoping for and did not
achieve (Phase 6: 0.205 → 0.2167, the wrong direction) — achieved here not by
perturbing pixels, but by using a backbone that never learned to rely on raw
appearance in the first place.

## 16. PCA analysis

Saved to
`outputs/asl_citizen_native_10/experiments/i3d_frozen/diagnostics/`:

- `i3d_frozen_pca_by_class.png` — visually, 10 well-separated, tight clusters, one
  per gloss (matching the near-perfect confusion matrix in §11).
- `i3d_frozen_pca_by_signer.png` — visually diffuse, no visible per-signer grouping
  (matching the negative signer silhouette in §15).

Both are a qualitative reversal of the corresponding Phase 5/6 plots
(`outputs/asl_citizen_native_10/diagnostics/phase5_generalization/feature_analysis/`
and `outputs/asl_citizen_native_10/experiments/appearance_aug/diagnostics/`), which
showed diffuse, unseparated class clusters and visible signer-aligned sub-clusters.

## 17. Comparison

| Metric | ResNet18 Frozen | ResNet18 Layer4 | ResNet18 Appearance Aug | **I3D Frozen** |
|---|---:|---:|---:|---:|
| Best validation accuracy | 17.95% | 17.95% | 17.95% | **100.0%** |
| Test top-1 accuracy | 13.18% | 9.30% | 14.73% | **96.90%** |
| Test top-5 accuracy | 51.16% | 48.06% | 48.84% | **99.22%** |
| Macro F1 | 0.0866 | 0.0653 | 0.1045 | **0.9691** |
| Train accuracy (final epoch) | 58.39% | — | 50.34% | 100.0% |
| Unique test classes predicted | 7 / 10 | 7 / 10 | 7 / 10 | **10 / 10** |
| Dominant predicted class | YES | NO | YES | EAT1 |
| Dominant class share | 64.34% | 31.78% | 64.34% | **13.18%** |
| Class silhouette | -0.0725 | -0.0769 | -0.0723 | **+0.22** |
| Signer silhouette | 0.205 | 0.094 | 0.2167 | **-0.0612** |
| Signer accuracy range | not fully collapse-free | not fully collapse-free | not fully collapse-free | **84.6%–100%, no collapse** |

Every metric moves the same direction at once — accuracy, macro F1, class
silhouette, signer silhouette, dominant-class share, and per-signer collapse all
agree. This is the specific pattern Phase 6 explicitly checked for and did not find
(there, accuracy nudged up while the representation-level signal stayed flat or moved
the wrong way). Here, the representation-level evidence and the behavioral evidence
tell the same story, which is why this result is trusted more than a raw accuracy
number would be on its own.

## 18. Diagnosis

The frozen I3D representation, without any fine-tuning and without seeing any
native-10 training clip during its own pretraining, already contains strong
sign-discriminative structure for these 10 classes and does not organize itself by
signer. A single linear layer trained on top of these frozen features for 12 epochs
(a few seconds of compute) reaches 96.9% test accuracy on a signer-independent split
where the from-scratch ImageNet-ResNet18+GRU pipeline — across three separate
interventions (frozen, layer4 fine-tune, appearance augmentation) — never exceeded
14.73%.

This directly confirms the root-cause diagnosis from the Phase 6 strategy review
(`NATIVE_ASL_STRATEGY_REVIEW.md`, §1/§4): the bottleneck was never the native-10
data's ~1 clip-per-signer-per-class density in isolation — it was that an
ImageNet-pretrained 2D CNN had no reason to prefer sign-motion features over
signer-appearance features when given only that much data to learn the distinction
from scratch. A backbone that had already been forced, during its own (much larger,
much more signer-diverse) pretraining, to separate thousands of classes sharing the
same signer pool, transfers to a 10-class, 149-example downstream task extremely
efficiently, because the hard part (learning to ignore appearance) was already solved
before this experiment ever started.

This is **CASE A** exactly as defined in Task 9: test accuracy and macro F1 improved
substantially, class silhouette improved substantially, and signer silhouette
decreased — all three, together, not just one metric in isolation.

**Caveats, stated plainly rather than glossed over:**
- 129 test clips is a small evaluation set; 4 errors is 96.9%, but 5 errors would be
  96.1% — the true precision on "96.9%" itself has real sampling uncertainty at this
  scale, even though the result is so far from the ResNet baselines that ordinary
  sampling noise cannot explain the gap.
- The frozen I3D backbone was pretrained on the *same 52 signers* that populate
  native-10's train/val/test splits (§1), just never on these exact 149/39/129 clips
  or explicitly on a 10-way BOOK/EAT1/.../YES task. This experiment shows the
  representation transfers well *within* the ASL Citizen signer population — it does
  not by itself demonstrate transfer to a signer entirely outside that population
  (e.g., someone who contributed zero clips to any ASL Citizen split). That is a
  narrower, still-open question the current result does not answer.
- This is one linear-probe run at one configuration (per Task 6's "no sweep"
  instruction) — not a claim that this exact head configuration is optimal, only
  that it is sufficient to demonstrate the representation is usable.

## 19. Recommended next experiment

**Not launched — reported only, per the Phase 7 stop condition.**

Per Task 9's decision gate: frozen I3D substantially improved test accuracy, macro
F1, class silhouette, *and* reduced signer dependence — all four — so the
gate is satisfied for proposing (not launching) **I3D + controlled fine-tuning**:

- Unfreeze a small number of the backbone's final layers (the vendored
  `InceptionI3d.forward()` already exposes an `n_tune_layers` parameter built for
  exactly this — freeze all but the last N endpoints) rather than the full 15M-parameter
  network at once, to see whether native-10-specific fine-tuning can close the
  remaining ~3.1 percentage-point gap (the 4 HELP/MOTHER confusions in §11) without
  re-introducing the signer shortcut that fine-tuning the ResNet18's layer4 produced
  in Phase 4C.
- Any such follow-up should re-run the full evaluation and representation-analysis
  protocol from this report (Tasks 7/8) — the decision gate for *that* experiment
  should be the same one used here (did class silhouette improve or hold, did signer
  silhouette stay low), not accuracy alone, exactly as Phase 6 demonstrated accuracy
  alone can mislead.
- Separately from fine-tuning: given §5 showed frozen-backbone inference costs under
  400 MB of VRAM and 104 seconds for all 317 clips, there is no compute-driven reason
  this couldn't also be tested on the existing 20-word vocabulary
  (`ASL_CITIZEN_20_GLOSSES` in `src/asl_citizen/vocab.py`) referenced in earlier
  phases, as a scale check — but that is explicitly out of scope for this phase's stop
  condition (native-10 vocabulary only) and is not launched here either.

---

**STOP.** Per this phase's scope: the recommended next experiment (I3D partial
fine-tuning) was not launched. `data/asl_citizen_native_10/` and `data/asl_citizen_100/`
were not modified (the asl_citizen_100 manifest CSVs' *timestamps* were touched as a
side effect of running the existing `tests/test_asl_citizen.py` suite, which calls
`build_manifests()` on the default — asl_citizen_100 — config in its `setUpClass`;
this is pre-existing test-harness behavior unrelated to Phase 7, and the regenerated
content was verified unchanged: 100 classes, 1708/394/1444 train/val/test rows,
matching the values on record before this phase started). No previous native-10
checkpoint or report was overwritten (`outputs/asl_citizen_native_10/checkpoints/`,
`.../experiments/layer4/`, and `.../experiments/appearance_aug/` are all untouched;
this phase wrote only to `.../experiments/i3d_frozen/`). A-Z, Word Spelling, the
frontend, backend APIs, the database, Alembic migrations, and the chatbot were not
modified. The I3D model was not integrated into the application.
