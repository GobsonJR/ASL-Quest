# ASL Citizen I3D checkpoint — provenance

This directory contains a **model checkpoint only**. It does not contain any
ASL Citizen dataset videos and must never be used to store them.

## Source

- Official repository: https://github.com/microsoft/ASL-citizen-code
  (Microsoft Research, author of record: `alexxijielu`)
- Release: `checkpoints_v1` — "ASL Citizen Weights 1.0"
  (https://github.com/microsoft/ASL-citizen-code/releases/tag/checkpoints_v1)
- Asset downloaded: `ASL_citizen_I3D_weights.zip`
  (56,382,633 bytes zipped; GitHub-reported release asset, verified via the
  GitHub Releases API before download)
- Extracted file: `ASL_citizen_I3D_weights.pt` (60,521,671 bytes,
  `torch.save`d `OrderedDict` state_dict — no wrapping metadata)
- SHA-256 of the extracted `.pt` file:
  `3538319620331d5ba731e0f9ac79161deeb37a895b7680d215864fc00a14477d`
- Downloaded and verified: 2026-09-22

## License

`LICENSE_microsoft_ASL-citizen-code.txt` in this directory is a verbatim
copy of the `LICENSE` file from the root of `microsoft/ASL-citizen-code` at
the time of download: **MIT License, Copyright (c) Microsoft Corporation.**

The repository's README publishes these checkpoint weights from the same
MIT-licensed project, in the same repository, with no separate license
notice attached to the release. They are treated here as covered by the
same MIT terms as the code repository that produces, documents, and
distributes them. This is a reasonable reading, not a legal certainty —
if a stricter reading is ever needed, treat the weights as "at least as
permissive as MIT, pending explicit confirmation from Microsoft."

This license covers the **code and model weights only**. It says nothing
about the ASL Citizen **dataset** (videos), which has its own separate
license/terms at https://www.microsoft.com/en-us/research/project/asl-citizen/
and must be requested/downloaded independently. **No dataset video is
stored in this repository or anywhere under `models/`.**

## Architecture verification

Loaded against `src/i3d_transfer/pytorch_i3d.py`'s `InceptionI3d` class
(vendored verbatim, MIT, from the same repository's `I3D/pytorch_i3d.py`,
itself sourced upstream from https://github.com/piergiaj/pytorch-i3d per
that file's own header comment):

```python
i3d = InceptionI3d(400, in_channels=3)
i3d.replace_logits(2731)
i3d.load_state_dict(torch.load("ASL_citizen_I3D_weights.pt"))
```

Result: **0 missing keys, 0 unexpected keys** (`strict=True` load
succeeded). 344 state_dict entries. 15,086,539 total parameters.
`logits.conv3d.weight` shape `[2731, 1024, 1, 1, 1]` confirms this
checkpoint's final layer was trained for the full 2,731-gloss ASL Citizen
vocabulary.

## Pretraining — corrected from the Phase 6 strategy review

The Phase 6 strategy review (`NATIVE_ASL_STRATEGY_REVIEW.md`) described
this checkpoint as "Kinetics-pretrained I3D, fine-tuned on ASL Citizen,"
based on secondary web-search summaries. **Direct inspection of the
official training script (`I3D/aslcitizen_training.py`) corrects this**:
the script instantiates `InceptionI3d(400, in_channels=3)` and calls
`replace_logits(2731)` immediately, with **no `load_state_dict()` call
before training** — i.e., no Kinetics checkpoint is loaded. The model is
trained **end-to-end from random initialization directly on the full ASL
Citizen training split** (Adam, lr=1e-3, up to 75 epochs, `ReduceLROnPlateau`
patience 5 factor 0.3, batch size 8) — not fine-tuned from a
Kinetics-pretrained starting point.

This is corrected here rather than left standing, because it changes what
this checkpoint's "pretraining domain" actually is: **the full ASL Citizen
corpus itself (all 2,731 glosses, all 52 signers, ~65k+ training clips)**,
not Kinetics-400 action recognition. That is if anything a *better* match
for our transfer purpose than a Kinetics-pretrained backbone would be — the
model was forced to become sign-discriminative and (by construction of a
2,731-way classification task where the same ~52 signers occur across
thousands of different classes) could not rely on a per-signer shortcut the
way our from-scratch native-10 ResNet18 could with only 10 classes and ~15
signers per class.

## Not included / not modified

- No ASL Citizen dataset video is present in this repository.
- `data/asl_citizen_native_10/` (the original manifest) was not modified;
  `data/asl_citizen_native_10_i3d/` holds a byte-identical copy of its
  train/val/test CSVs and class mappings for this isolated experiment.
