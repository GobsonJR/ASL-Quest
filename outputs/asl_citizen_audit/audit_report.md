# ASL Citizen Dataset Audit Report (Phase 7A)

Generated: 2026-09-14T22:46:37.315512+00:00
Dataset root: `D:\ASL_Citizen\ASL_Citizen`

## 1. Dataset Structure

```
D:\ASL_Citizen\
└── ASL_Citizen\
    ├── splits\
    │   ├── train.csv
    │   ├── val.csv
    │   └── test.csv
    ├── videos\          (flat .mp4 files)
    └── use.txt            (Microsoft Research license)
```

- Total files: **83403**
- Total directories: **2**
- Video files: **83399**
- CSV/metadata files: **3** CSV, **0** JSON
- Disk usage: **46.198 GB**
- Video formats: **{'.mp4': 83399, '.csv': 3, '.txt': 1}**

## 2. Label / Metadata

CSV columns: `Participant ID`, `Video file`, `Gloss`, `ASL-LEX Code`.

A video maps to a sign via the **Gloss** column. Signer is **Participant ID** (P1..P52).
Video ID is the numeric prefix before the first `-` in the filename.

### Example rows

- `15890366051589533-APPLE.mp4` → gloss **APPLE**, signer **P1**, ASL-LEX **A_03_054**, split **train**
- `35618482303951104-IMPOSSIBLE.mp4` → gloss **IMPOSSIBLE**, signer **P1**, ASL-LEX **B_01_032**, split **train**
- `6958143575951994-PARK.mp4` → gloss **PARK**, signer **P1**, ASL-LEX **E_03_028**, split **train**
- `8006032738002744-SOCCER 2.mp4` → gloss **SOCCER2**, signer **P1**, ASL-LEX **F_03_032**, split **train**
- `37542279833186454-STINK.mp4` → gloss **STINK**, signer **P1**, ASL-LEX **H_01_064**, split **train**

## 3. Class Distribution

- Distinct signs/classes: **2731**
- Total videos (metadata): **83399**
- Videos/class — min: **21**, max: **45**, mean: **30.5379**, median: **31**

### Top 30 classes by video count

1. **DOG1** — 45 videos
2. **BASKETBALL1** — 44 videos
3. **WHATFOR1** — 43 videos
4. **BELT1** — 40 videos
5. **BEE1** — 39 videos
6. **BITE1** — 39 videos
7. **BREAKFAST1** — 39 videos
8. **CHRISTMAS1** — 39 videos
9. **DARK1** — 39 videos
10. **DEAF1** — 39 videos
11. **DEMAND1** — 39 videos
12. **EAT1** — 39 videos
13. **ELEVATOR1** — 39 videos
14. **FINE1** — 39 videos
15. **FOREIGNER1** — 39 videos
16. **HOSPITAL1** — 39 videos
17. **MECHANIC1** — 39 videos
18. **MOVIE1** — 39 videos
19. **NIGHT1** — 39 videos
20. **PARTY1** — 39 videos
21. **PATIENT2** — 39 videos
22. **ROCKINGCHAIR1** — 39 videos
23. **SHAVE1** — 39 videos
24. **AXE1** — 38 videos
25. **BELIEVE1** — 38 videos
26. **DECIDE1** — 38 videos
27. **DOWNSIZE1** — 38 videos
28. **DRAG1** — 38 videos
29. **EDIT1** — 38 videos
30. **HURDLE/TRIP1** — 38 videos

- Classes with ≤5 videos: **0**
- Classes with ≥20 videos: **2731**

## 4. Official Split

| Split | Videos | Classes | Participants |
|-------|--------|---------|--------------|
| train | 40154 | 2731 | 35 |
| val   | 10304 | 2731 | 6 |
| test  | 32941 | 2731 | 11 |

## 5. Signer Split Analysis

- Signer-independent: **True**
- Train∩Val overlap: **0** participants
- Train∩Test overlap: **0** participants
- Val∩Test overlap: **0** participants
- Note: Official ASL Citizen split is signer-independent: each participant appears in exactly one split.

## 6. Video Quality (sampled)

- Sample size: 300
- Decodable: 300 / Undecodable: 0
- Duration stats (sec): {'count': 300, 'min': 0.869, 'max': 8.8, 'mean': 2.9472, 'median': 2.667}
- Frame count stats: {'count': 300, 'min': 20.0, 'max': 264.0, 'mean': 86.8667, 'median': 79.0}
- FPS distribution (sample): {30.0: 165, 29.9: 35, 30.4: 19, 30.3: 11, 25.0: 10, 29.8: 9, 30.5: 7, 15.0: 6, 30.6: 6, 31.0: 5, 30.1: 3, 30.8: 3, 15.2: 2, 29.7: 2, 30.7: 2, 31.1: 2, 13.3: 1, 24.2: 1, 25.1: 1, 25.6: 1, 28.4: 1, 28.5: 1, 28.6: 1, 28.7: 1, 28.8: 1, 29.3: 1, 30.2: 1, 30.9: 1, 31.4: 1}
- Resolution distribution (sample): {'640x480': 293, '960x540': 7}

## 7. Duplicates / Leakage

- Duplicate filenames in metadata: 0
- Duplicate video IDs in metadata: 0
- Same filename across splits: 0
- Missing CSV-referenced files on disk: 0
- Partial-hash identical groups (500-file sample): 0

## 8. Static vs Dynamic Signs

ASL Citizen is an isolated-sign video dataset. Many signs require movement/path (e.g., signs with motion paths, two-part compounds). Single-frame classification is insufficient for a large portion of the vocabulary; temporal models (sampled frames + RNN/Transformer) are appropriate.

## 9. Image vs Video Representation

Recommended: **video → uniformly sampled temporal frames → spatial encoder → temporal model → class**.
Matches existing ResNet18+GRU pipeline and dataset characteristics (short clips, movement, idle padding).

## 10. Candidate Subsets

- **50 classes**: 1927 videos, avg 38.54/class, min 37, train/val/test 933/212/782
- **100 classes**: 3544 videos, avg 35.44/class, min 32, train/val/test 1707/393/1444
- **200 classes**: 6744 videos, avg 33.72/class, min 32, train/val/test 3203/760/2781
- **300 classes**: 9885 videos, avg 32.95/class, min 31, train/val/test 4699/1128/4058

**Recommendation:** Start with **100 classes** for Phase 7B — strong per-class support, manageable training time on RTX 4050 6GB, and useful vocabulary size for a teaching platform.

## 11. Storage

- Dataset on disk: 46.198 GB
- D: free space: 175.37 GB
- Estimated extras: {'manifests_and_metadata': 0.05, 'optional_cached_frames_16f_224_subset100': 8, 'optional_cached_frames_16f_224_full2731': 120, 'model_checkpoints_resnet18_gru': 0.15, 'training_outputs_logs': 0.5}

## 12. Realistic Accuracy Expectations (vs WLASL100)

ASL Citizen is **more suitable** than WLASL100 for a teaching platform in several ways:
- **3× more examples per class** (mean ~30.5 vs WLASL100 ~10–20 typical)
- **Signer-independent official split** (35/6/11 participants) — realistic generalization test
- **Crowdsourced webcam data** — closer to student webcam use than lab-recorded WLASL
- **All 2,731 classes appear in every split** — no WLASL100-style missing test classes

Challenges vs WLASL100:
- **27× larger vocabulary** if training on full set — much harder multi-class problem
- **Background/lighting/camera variation** is higher (self-recorded, non-studio)
- **No frame_start/frame_end annotations** — must use full clip or detect signing span
- Published I3D baseline: **63% top-1**, **91% recall@10** on full 2,731-class signer-independent test

For a **100-class subset** on ResNet18+GRU (lighter than I3D), realistic expectations:
- **Top-1:** moderate — likely below paper's 63% full-vocab I3D, but above typical small-data WLASL runs
- **Top-5 / recall@10:** more promising for dictionary-style retrieval
- Main accuracy drivers: subset size, temporal modeling, signing-span trimming, signer diversity at test time

## 13. Word Model Code Reuse (`src/word_model/`)

| Module | Reusable? | WLASL-specific assumptions to replace |
|--------|-----------|---------------------------------------|
| `model.py` | **Yes** | Default `num_classes=100`; architecture itself is dataset-agnostic |
| `dataset.py` | **Partial** | `WLASL100VideoDataset`, manifest `frame_start`/`frame_end`, WLASL paths |
| `train.py` | **Mostly** | CLI descriptions, dataloader wiring to WLASL manifest |
| `evaluate.py` | **Mostly** | WLASL titles/paths; evaluation loop is reusable |
| `predict.py` | **Mostly** | WLASL checkpoint defaults; `frame_start`/`frame_end` optional args |
| `checkpoint.py` | **Yes** | Docstrings only; format is generic |
| `metrics.py` | **Yes** | WLASL100 missing-test-class note in `compute_evaluation_metrics` |
| `training_utils.py` | **Yes** | Fully generic |
| `utils.py` | **No** | Hard-coded 100-class list, WLASL manifest columns/paths |
| `config.py` | **No** | All WLASL100 paths, `NUM_CLASSES=100`, output dirs |

**Phase 7B redesign:** new `config.py`/`utils.py`, manifest builder from ASL Citizen CSVs, dataset class rename + optional signing-span detection (no NSLT frame annotations).

## 14. Safety

- No training performed.
- No protected project assets modified (`models/`, `frontend/`, `backend/`, `data/asl_quest.db`, A–Z pipeline).
- No dataset files moved/deleted.
- Project tests: **57/57 passed** (`python -m unittest discover -s tests -v`).
