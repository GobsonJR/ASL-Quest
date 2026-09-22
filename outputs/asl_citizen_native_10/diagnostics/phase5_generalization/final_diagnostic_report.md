# Native-10 Phase 5 Generalization Diagnostic

No models were trained in this phase. All analysis below uses the existing Phase 4B (frozen) and Phase 4C (layer4) checkpoints for inference/feature-extraction only, the existing native_10 manifest (unmodified), and read-only inspection of the 317 dataset videos. Diagnostic script: `scripts/diagnose_native_10_phase5.py`. Full outputs: `outputs/asl_citizen_native_10/diagnostics/phase5_generalization/`.

## 1. Executive findings

The strongest, most consistently supported finding, corroborated by three independent methods (feature-space clustering, PCA visualization, and per-signer prediction breakdown), is that **both checkpoints organize their learned representation around signer identity more than sign identity, and in several individual signers' cases predict almost the same 1-2 classes for that signer regardless of which of the 10 signs they actually performed.** This is enabled by a concrete dataset property: native_10 training clips average only ~1.0-1.3 clips per signer per class, so each class is defined by an almost-disjoint set of individual people, leaving a signer-correlated shortcut available and apparently exploited. Neither the current temporal sampling strategy nor the spatial crop appears to be a meaningful bottleneck (see §5, §6). The layer4 epoch-1 NaN gradient has a well-supported, reproducible mechanistic explanation (small-batch + fp16 autocast interaction) but no evidence ties it to the shared generalization failure, since the frozen run failed similarly with no NaN at all.

## 2. Dataset distribution

317 total clips (149 train / 39 val / 129 test). Predominantly 640×480 at ~30fps; mean clip duration 2.67s (σ=1.62, range 1.0-22.1s), mean frame count 78.9 (σ=48.8, range 17-664). Per-class train counts range 13-18 (mean 14.9, σ=1.37); per-class train signer counts range 12-16 (mean 14.2, σ=1.17) — i.e. class size and signer count per class are nearly identical, confirming almost every training clip in a class comes from a different person.

**Concrete resolution split difference:** 13 of 317 clips (4.1% overall) are 960×540 (16:9) rather than 640×480 (4:3). All 13 fall in the **test split only** — zero in train, zero in val — spread across 9 of the 10 classes (1-3 per class). Train and validation never see this camera's aspect ratio/framing; roughly 10% of the test split (13/129) does.

## 3. Train/validation/test differences

Per-class mean clip duration is broadly comparable across splits (train/val/test means mostly within ~2.2-3.4s of each other per class), with one exception: YES's train clips average 3.92s vs 2.62s (val) and 2.52s (test) — the largest single split gap observed, though based on only 15/5/10 clips. Signer sets are fully disjoint across splits by construction (re-verified, no participant overlap). The resolution/aspect-ratio gap in §2 is the clearest quantified train-vs-test distribution difference found.

## 4. Visual audit

Contact sheets for all 10 classes (begin/mid/end frames × train/val/test) saved under `visual_audit/`. Manual review of the YES sheet (the frozen model's dominant collapse class) shows three visibly different recording setups across splits for the same gloss: the train example is notably under-lit with the signer positioned farther from a small home-office-style camera (bookshelf/blinds background); the validation example is brightly and evenly lit with the signer close to camera against a plain mint-green wall with a framed picture; the test example is bright with a colorful, cluttered patterned background (toys/decorations) and a different signer entirely. Camera distance, lighting, background clutter, and signer appearance all visibly differ between the three clips of the same class. This is direct visual confirmation that strong appearance confounds exist per-video, consistent with §7's signer-collapse finding.

## 5. Temporal sampling analysis

Four strategies compared on ~20 sample train clips (statistics only, no training): current uniform sampling achieves full (100%) coverage of each video's span with mean spacing 5.08 frames; a 2x-denser variant halves the spacing (2.46 frames) but by construction still spans the same duration when the video is long, or uses every frame when short; a center-focused variant intentionally covers only ~51% of the span (mean spacing 2.57 frames); a start/mid/end segmented variant produces near-identical coverage/spacing to the current strategy. On a 5-video motion-energy subset (mean absolute pixel difference between consecutive sampled frames), the current uniform strategy captured motion energy equal to or greater than the denser and center-focused alternatives in 4/5 videos. **No evidence that the current sampling strategy is missing important motion relative to the alternatives tested.**

## 6. Spatial preprocessing analysis

Train-time `RandomResizedCrop` retains 85-100% of the resized-frame area by config (`scale=[0.85, 1.0]`); measured on 6 real samples across classes/splits, retained area was 87.1-91.5%. Eval center-crop retains a fixed 87.1%. Visual inspection of the generated before/after grids (e.g. `HELLO_test_preprocessing.png`) confirms the signing hand and its raised position remain fully visible after both the train and eval crop in the reviewed examples. **No evidence the crop removes or obscures the signing hand;** this pipeline has no hand/pose landmarks (unlike the production A-Z MediaPipe path), so this conclusion rests on the crop-area statistics plus manual visual review of the saved images, not automated hand-visibility measurement.

## 7. Feature representation analysis

Pooled GRU features (256-d, pre-classifier) extracted for all 129 test clips from both checkpoints (inference only, no gradient updates).

| Metric | Frozen | Layer4 |
|---|---|---|
| Silhouette score, grouped by **true class** | **-0.0725** | **-0.0769** |
| Silhouette score, grouped by **signer** | **+0.205** | **+0.094** |
| Nearest-centroid accuracy* | 31.01% | 22.48% |
| Trained classifier top-1 (cross-check) | 13.18% | 9.30% |
| Inter-class centroid mean distance | 1.13 | 1.93 |

*Nearest-centroid accuracy uses class centroids computed from this same test set, so each sample partly informs its own class's centroid — an optimistic, circular upper bound on separability, not a valid held-out metric. It is reported to show that even under this best-case, non-held-out framing, class separability is weak (31%/22%, far below tiny-overfit's 96-98%) — and that the trained softmax classifier scores lower still (13%/9%) than this optimistic bound, suggesting the trained head may not even be fully exploiting the modest class signal the features do carry.

Both checkpoints show **negative** silhouette by true class (features do not cluster by sign at all — worse than a random partition) and **positive** silhouette by signer (features cluster more by who is signing). PCA 2D projections (`feature_analysis/*_pca_by_class.png` vs `*_pca_by_signer.png`) visually corroborate this: coloring by class shows uniformly intermixed points with no visible separation, while coloring by signer shows several individually-colored signer clusters occupying distinct regions of the projection (frozen checkpoint, PCA explained variance 25.5%+14.3%). This is not overinterpreted from the 2D projection alone — it is consistent with the silhouette scores computed in the full 256-d feature space.

## 8. Signer analysis

Per-signer test predictions (`signer_analysis.json`) show pronounced collapse to 1-2 classes regardless of the true sign performed:

- Frozen: signer P49 — 10/10 clips predicted "YES" (true labels varied); P18 — 7/11 predicted "HELP"; P42 — 10/13 predicted "YES"; P9 — 11/14 predicted "YES".
- Layer4: signer P35 — 11/14 predicted "EAT1"; P47 — 8/11 predicted "NO"; P18 — 7/11 predicted "WATER".

If the model were using sign-discriminative features, a given signer's predicted-class distribution (across their several *different* true signs) should roughly track their true-label distribution, not collapse onto one or two fixed classes. This pattern — collapse per **person**, not per **sign** — is the clearest sample-level evidence for the appearance-shortcut hypothesis, and its strength (frozen more extreme than layer4) matches §7's silhouette-by-signer gap (0.205 vs 0.094).

## 9. Confusion analysis

From the existing confusion matrices (both experiments, not re-run): frozen collapses almost entirely onto **YES** — every one of the other 9 classes' single most common misprediction is YES (e.g. HELP→YES 11/13, EAT1→YES 10/17, HELLO→YES 9/14), while the reverse (YES mispredicted as something else) essentially never occurs — a strongly asymmetric, single-attractor pattern rather than confusion between visually similar signs. Layer4 funnels errors into a small "NO / WATER / YES" attractor set instead (EAT1→NO 6/17, HELLO→NO 6/14, MOTHER→NO or WATER 5+4/15, HELP→NO or WATER 4+4/13), again strongly asymmetric. Layer4 never predicted BOOK, HELLO, or PLEASE on any of the 129 test clips. Neither pattern lines up with visually-similar sign pairs (e.g. no particular concentration between signs sharing handshape/location); both look like collapse onto a small number of "default" output classes.

## 10. Augmentation analysis

The tiny-overfit gate (96% frozen / 98% layer4, both passing) used fully deterministic `eval_spatial_transforms` (no `RandomResizedCrop`, `temporal_jitter=0`) for **both** backbone modes. Real training adds: (a) `RandomResizedCrop` retaining 85-100% of the resized frame (mild, and per §6 does not remove the hand), and (b) ±2-frame temporal jitter, which is **36.8%** of the mean 5.43-frame spacing between the 16 uniformly-sampled frames in native_10 train clips — a non-trivial fraction, meaning jitter can plausibly shift which visual instant is sampled by close to one full "slot." Given the crop is mild and visually confirmed not to remove the hand, spatial augmentation alone is an unlikely sole explanation for the 96%→13% gap; the temporal jitter magnitude is a more plausible, still-unverified contributing factor. No training was run to isolate this further, per the phase's constraints.

## 11. Layer4 NaN analysis

Training-log evidence: `training_history.json`'s epoch-1 gradient snapshot (taken once, after that epoch's *last* batch) is NaN; epoch 2 onward is finite. The dataset has 149 train clips at batch_size=4 with `drop_last=False`, so **every** epoch's final batch has exactly 1 sample (149 = 37×4 + 1).

Mechanistic probe (3 independent trials, fresh pretrained layer4 model each time, single-batch forward+backward, no optimizer step, no checkpoint changes — mirroring the existing `07_optimization_gradients.json` diagnostic convention used for the 100-class layer4 failure): a batch of size 1 run under AMP (float16 autocast + GradScaler) produced a **non-finite layer4 gradient in 3/3 trials**. The identical batch content run without AMP (fp32) was finite in 3/3 trials, and a batch of size 4 under AMP was finite in 3/3 trials. Inputs, encoder outputs, GRU outputs, logits, and loss were all finite at every stage in every trial — the non-finite value appeared specifically during the AMP backward pass on the size-1 batch.

**Supported:** the small-batch (size 1) + fp16-autocast interaction reliably produces non-finite layer4 gradients on a freshly-initialized model, and the training's epoch structure guarantees a size-1 batch every epoch — a well-supported mechanistic explanation for the epoch-1 event. **Plausible but unverified:** why epochs 2-25 (which also end on a size-1 batch) did not recur in the logged snapshot — PyTorch's `GradScaler` halves its scale factor after detecting an inf/nan, which would plausibly prevent recurrence once triggered once, but per-batch scale history was not logged during the real run, so this is not certified. **Unsupported:** that this NaN caused or contributed to the poor generalization — training self-corrected by epoch 2, and the frozen run (zero NaN events) reached an equally poor (arguably more collapsed) generalization outcome, so NaN is not needed to explain the shared failure mode.

## 12. Dataset-size limitations

149 train clips / 10 classes (mean 14.9/class) with 12-16 unique signers per class (mean 14.2) means **mean clips-per-signer-per-class is 1.0-1.3 across every one of the 10 classes** — almost every training example of a given sign comes from a person the model will never see sign anything else. This removes the natural averaging effect (same class, many different appearances) that would otherwise force the model to discard signer-specific cues. This is a specific, verified statistical mechanism — not just "the dataset is small" — and it lines up exactly with what §7/§8 found the model actually learned. The limiting factor is signer diversity per class, not raw clip count: more clips from the *same* already-seen signers would not address it. What would help is more clips of each existing gloss from **additional signers not already in the training set**, increasing signer-diversity-per-class rather than clip count alone.

## 13. Most strongly supported causes

| Suspected cause | Classification |
|---|---|
| Model represents signer identity/appearance more than sign identity | **Supported by evidence** (negative class-silhouette + positive signer-silhouette in both checkpoints, PCA visual confirmation, per-signer prediction collapse to 1-2 classes regardless of true sign) |
| ~1.0-1.3 clips/signer/class in training provides an exploitable signer-correlated shortcut | **Supported by evidence** (direct manifest counts) |
| Visible per-video confounds (background, lighting, camera distance, clothing) exist across signers/splits | **Supported by evidence** (visual contact-sheet review) |
| Small-batch (size 1) + fp16 AMP causes non-finite layer4 gradients | **Supported by evidence** (3/3 reproducible single-batch trials) |
| Test-split resolution/aspect-ratio shift (13 clips, test-only) affects test accuracy | **Supported by evidence** that the shift exists; **plausible but unverified** that it materially affects the reported accuracy, given it touches only ~10% of test |
| Trained softmax classifier under-exploits the (weak) class signal present in the features | **Plausible but unverified** (nearest-centroid beats the trained classifier, but the comparison is methodologically optimistic/circular) |
| ±2-frame temporal jitter (37% of mean sample spacing) meaningfully perturbs training signal vs tiny-overfit's jitter-free setup | **Plausible but unverified** (magnitude is non-trivial; not isolated by an ablation in this read-only phase) |
| RandomResizedCrop spatial augmentation gap vs tiny-overfit's deterministic crop | **Plausible but unverified**, and likely a minor contributor at most (crop retains 85-91% of area; hand visibly intact) |
| Current uniform temporal sampling misses important sign motion | **Unsupported** (full video-span coverage; motion-energy proxy comparable to or better than alternatives tested) |
| Spatial cropping removes/obscures the signing hand | **Unsupported** (mild crop area; hand visibly intact in reviewed samples) |
| Epoch-1 NaN gradient caused or contributed to the generalization failure | **Unsupported** (frozen run had zero NaN events and an equally poor, arguably more collapsed, outcome) |

## 14. Recommended next experiment

**One experiment, not launched:** train a single frozen-backbone model, identical to Phase 4B in every respect (architecture, split, vocabulary, seed, epochs/patience, optimizer) except for adding appearance-decorrelating spatial augmentation (color/brightness/contrast jitter, applied consistently across each clip's 16 frames the same way the existing consistent-crop logic is applied) on top of the existing pipeline — then run the exact same diagnostic-5/6 analysis (silhouette-by-class vs silhouette-by-signer, PCA projection, per-signer prediction-collapse check) on its checkpoint. This experiment directly targets the strongest and most consistently supported finding in this report (§7, §8, §12: the representation organizes by signer, not sign, because the dataset offers almost no same-class/different-signer repetition) and asks one clear, falsifiable question: does appearance-decorrelating augmentation measurably shift the learned representation toward class-discriminative structure (higher class silhouette, lower signer silhouette, less per-signer prediction collapse) without any change to vocabulary, split, or architecture? It does not require new data acquisition, does not touch production, and does not expand the class vocabulary.

**STOP.** This experiment is a recommendation only and was not run. No training occurred in this phase; production systems (A-Z model, Word Spelling, frontend, database, Alembic migrations, chatbot, asl_citizen_100, and the native_10 manifest) were not touched.
