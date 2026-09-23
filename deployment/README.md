# Transfer package — runtime assets that live outside Git

Everything ASL-Quest needs to **run** falls into two groups:

1. **Source code, migrations, and small manifests** — already in this Git
   repository. `git clone` (or copying the whole repo folder) is enough.
2. **Large/binary/licensed runtime assets** — deliberately **not** committed
   (see `.gitignore`: `*.pth`, `*.pt`, `*.task`, `data/asl_quest.db`). These
   must be copied by hand from a machine that already has them.

This folder is a **staging area** for group 2, not a copy of the assets
themselves (nothing here is committed except this README and the two
`.gitkeep` placeholders). The workflow is:

```
1. On the OLD machine, copy the files listed below into deployment/models/
   and (optionally) deployment/database/ on the NEW machine — via USB
   drive, network share, SCP, etc. (see "How to copy" below).
2. Run scripts\windows\install_deployment_assets.ps1
   — it moves each file from deployment/ to its real destination path
   and refuses to silently overwrite anything already there.
3. Delete the files from deployment/models/ and deployment/database/
   once installed (the script does this for you after a successful copy).
```

You can also skip this staging step entirely and copy files directly to
their destination paths yourself — the table below is the source of truth
either way.

## Required model files → `deployment/models/`

| # | File (drop into `deployment/models/`) | Real destination | Approx. size | Required for | Git status |
|---|---|---|---|---|---|
| 1 | `best_model.pth` | `models/best_model.pth` | ~43 MB | A-Z recognition (production checkpoint) | Ignored (`*.pth`) — must copy |
| 2 | `hand_landmarker.task` | `models/hand_landmarker.task` | ~7.5 MB | A-Z hand cropping (MediaPipe) | Ignored (`*.task`) — must copy |
| 3 | `ASL_citizen_I3D_weights.pt` | `models/pretrained/asl_citizen_i3d/ASL_citizen_I3D_weights.pt` | ~58 MB | Native Signs (frozen I3D backbone) | Ignored (`*.pt`) — must copy |
| 4 | `head_best.pt` | `outputs/asl_citizen_native_10/experiments/i3d_frozen/checkpoints/head_best.pt` | ~43 KB | Native Signs (trained linear head) | Ignored (`*.pt`) — must copy |

`install_deployment_assets.ps1` matches by filename, so just drop all four
files (flat, no subfolders) into `deployment/models/`.

### Where each one can come from if you don't have a copy

- **`best_model.pth`** — no public download; it's this project's own trained
  checkpoint. Either copy it from a machine that has it, or retrain it with
  `train.py` (requires the training image set under `dataset/asl_alphabet_train/`,
  which is a separate, large, non-restricted dataset not itself covered by
  this project's documentation — verify its license/terms before using it if
  you don't already have a copy from the original source).
- **`hand_landmarker.task`** — an official Google MediaPipe asset. Copy it
  from an existing machine, or download the current release from Google's
  own Hand Landmarker documentation:
  https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
  (Models section) and save it as `models/hand_landmarker.task`.
- **`ASL_citizen_I3D_weights.pt`** — an MIT-licensed checkpoint published by
  Microsoft Research. See `models/pretrained/asl_citizen_i3d/SOURCE.md` for
  the exact release, asset name, and SHA-256 checksum: download
  `ASL_citizen_I3D_weights.zip` from
  https://github.com/microsoft/ASL-citizen-code/releases/tag/checkpoints_v1
  and extract `ASL_citizen_I3D_weights.pt` from it.
- **`head_best.pt`** — this project's own trained linear head for the
  native-10 vocabulary. No public download; copy it from an existing
  machine, or retrain it with `scripts/train_native_10_i3d_frozen.py`
  (requires licensed ASL Citizen dataset access — see below).

## Optional: demo database → `deployment/database/`

Only needed if you're deliberately transferring the **existing** SQLite
database (with its current users/progress) instead of starting from a fresh,
migrated + seeded database. See `SETUP_WINDOWS.md` for the recommendation
and reasoning — the default recommendation is a **fresh database**, not this.

| File (drop into `deployment/database/`) | Real destination |
|---|---|
| `asl_quest.db` | `data/asl_quest.db` |

`install_deployment_assets.ps1` will refuse to overwrite an existing,
non-empty `data/asl_quest.db` unless you pass `-Force`.

## Restricted dataset — explicitly NOT part of this package

The **ASL Citizen video dataset** (Microsoft Research, the source videos
behind `models/pretrained/asl_citizen_i3d/ASL_citizen_I3D_weights.pt` and the
native-10 training clips) is **never** included here, in Git, or anywhere
else in this repository. It is not required to *run* the app — only to
*retrain or re-evaluate* the Native Signs model or the underlying ASL
Citizen pipeline.

If a team member genuinely needs the dataset for training/research, they
must request/obtain it directly from Microsoft Research under its own terms:
https://www.microsoft.com/en-us/research/project/asl-citizen/ — this project
does not grant, sublicense, or redistribute access to it. Once obtained
locally, point `ASL_CITIZEN_DATASET_ROOT` (see `.env.example`) at it; nothing
else in the repo needs to change.
