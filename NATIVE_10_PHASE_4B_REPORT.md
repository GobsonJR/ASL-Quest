# Native-10 Phase 4B Report

## 1. Pre-flight

The existing native-10 wrapper performed explicit pre-flight checks before the
recorded run:

- Dataset: `D:\SIGN LANGUAGE\data\asl_citizen_native_10\`
- Output root: `D:\SIGN LANGUAGE\outputs\asl_citizen_native_10\`
- Best checkpoint:
  `D:\SIGN LANGUAGE\outputs\asl_citizen_native_10\checkpoints\asl_citizen_native_10_resnet18_gru_best.pth`
- Last checkpoint:
  `D:\SIGN LANGUAGE\outputs\asl_citizen_native_10\checkpoints\asl_citizen_native_10_resnet18_gru_last.pth`
- No training output path resolved to `outputs/asl_citizen_100/`.
- Class mapping was verified as native-10 and alphabetically indexed.
- Split counts were verified as train `149`, validation `39`, and test `129`.
- Official signer split was verified with no train/validation, train/test, or
  validation/test participant overlap.
- The model state reloaded with no missing or unexpected keys.

Class mapping:

| Index | Gloss |
|---:|---|
| 0 | BOOK |
| 1 | EAT1 |
| 2 | HELLO |
| 3 | HELP |
| 4 | MOTHER |
| 5 | NO |
| 6 | PLEASE |
| 7 | THANKYOU |
| 8 | WATER |
| 9 | YES |

## 2. Configuration

| Setting | Value |
|---|---|
| Dataset | `data/asl_citizen_native_10/` |
| Architecture | Existing ResNet18 + GRU |
| Backbone mode | Frozen |
| Head learning rate | `0.001` |
| Backbone learning rate | `0.0001` |
| Dropout | `0.3` |
| Frames per clip | `16` |
| Batch size | `4` |
| Data-loader workers | `2` |
| Optimizer | AdamW |
| Seed | `42` |
| Maximum epochs | `30` |
| Early-stopping patience | `12` |
| Weight decay | Existing configuration (`1e-4`) |
| Trainable parameters | `593,930` |
| Frozen parameters | `11,176,512` |
| Total parameters | `11,770,442` |

The same native-10 wrapper preserved the existing spatial transforms,
validation transforms, temporal sampling, temporal jitter, normalization, and
class mapping. No dataset, manifest, split, seed, architecture, augmentation,
or production system was changed.

## 3. Training history

The run completed 23 of the 30 maximum epochs because early stopping triggered
after 12 consecutive epochs without validation-accuracy improvement. The
learning rate remained `0.001` for the head throughout the run.

| Epoch | Train loss | Train acc. | Val loss | Val acc. | Val top-5 | Unique val predictions |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.666678 | 8.72% | 2.362755 | 10.26% | 48.72% | 2 |
| 2 | 2.401767 | 10.74% | 2.439503 | 7.69% | 48.72% | 2 |
| 3 | 2.368321 | 8.72% | 2.338216 | 7.69% | 46.15% | 2 |
| 4 | 2.340329 | 16.78% | 2.284906 | 7.69% | 66.67% | 5 |
| 5 | 2.324553 | 10.07% | 2.302384 | 12.82% | 56.41% | 5 |
| 6 | 2.211318 | 17.45% | 2.349534 | 5.13% | 53.85% | 4 |
| 7 | 2.237501 | 18.12% | 2.302158 | 7.69% | 58.97% | 6 |
| 8 | 2.142873 | 19.46% | 2.341772 | 12.82% | 53.85% | 4 |
| 9 | 2.090542 | 25.50% | 2.326648 | 12.82% | 53.85% | 6 |
| 10 | 2.075310 | 22.15% | 2.511794 | 15.38% | 43.59% | 4 |
| 11 | 1.950838 | 28.86% | 2.408278 | 17.95% | 48.72% | 4 |
| 12 | 1.949117 | 26.85% | 2.361553 | 17.95% | 51.28% | 5 |
| 13 | 1.899080 | 30.20% | 2.578876 | 17.95% | 46.15% | 5 |
| 14 | 1.717872 | 41.61% | 2.476788 | 12.82% | 48.72% | 5 |
| 15 | 1.716576 | 37.58% | 2.488431 | 5.13% | 41.03% | 6 |
| 16 | 1.597913 | 41.61% | 2.664275 | 7.69% | 51.28% | 4 |
| 17 | 1.619963 | 38.26% | 2.485978 | 15.38% | 51.28% | 7 |
| 18 | 1.496464 | 46.98% | 2.692333 | 10.26% | 43.59% | 5 |
| 19 | 1.434957 | 48.99% | 2.578163 | 7.69% | 56.41% | 8 |
| 20 | 1.340042 | 48.99% | 2.571903 | 7.69% | 58.97% | 6 |
| 21 | 1.233069 | 57.05% | 2.736879 | 7.69% | 56.41% | 8 |
| 22 | 1.243908 | 55.70% | 2.651718 | 15.38% | 58.97% | 8 |
| 23 | 1.226280 | 58.39% | 2.553185 | 12.82% | 58.97% | 9 |

## 4. Best checkpoint

- Best checkpoint epoch: **11**
- Best validation accuracy: **17.95%**
- Best validation top-5 accuracy: **48.72%**
- Best checkpoint exists at the native-10 output path.
- Checkpoint architecture: `resnet18_gru`
- Checkpoint class count: `10`
- Checkpoint class mapping matches the native-10 manifest: **yes**
- State-dict missing keys: **none**
- State-dict unexpected keys: **none**
- Reloaded logits are finite: **yes**

## 5. Validation results

The best validation result was epoch 11 at **17.95% top-1 accuracy**. This is
above the 10% ten-class chance level, but it is still weak and did not improve
after epoch 11. The highest recorded validation prediction diversity was nine
of ten classes at epoch 23.

## 6. Test results

The official test split was evaluated once using only the best checkpoint.

| Metric | Result |
|---|---:|
| Samples | 129 |
| Accuracy / top-1 | **13.18%** |
| Top-5 accuracy | **51.16%** |
| Macro precision | **0.0942** |
| Macro recall | **0.1446** |
| Macro F1 | **0.0866** |
| Unique predicted classes | **7 / 10** |
| Dominant predicted class | **YES** |
| Dominant class share | **64.34%** |
| Logits finite | **yes** |

## 7. Confusion matrix

Rows are true labels and columns are predicted labels. Column/row order is:
`BOOK, EAT1, HELLO, HELP, MOTHER, NO, PLEASE, THANKYOU, WATER, YES`.

```text
[
  [1, 1, 0, 0, 1, 0, 0, 1, 0, 8],
  [1, 4, 0, 0, 0, 0, 0, 2, 0, 10],
  [0, 2, 0, 2, 0, 0, 0, 1, 0, 9],
  [0, 1, 0, 1, 0, 0, 0, 0, 0, 11],
  [0, 4, 0, 1, 0, 0, 0, 3, 0, 7],
  [1, 1, 0, 0, 0, 0, 0, 1, 0, 8],
  [0, 1, 1, 0, 0, 0, 0, 2, 0, 9],
  [1, 1, 1, 1, 0, 0, 0, 3, 0, 5],
  [0, 3, 0, 1, 0, 0, 0, 0, 0, 8],
  [0, 0, 0, 1, 1, 0, 0, 0, 0, 8]
]
```

## 8. Prediction distribution

| Predicted gloss | Count |
|---|---:|
| BOOK | 4 |
| EAT1 | 18 |
| HELLO | 2 |
| HELP | 7 |
| MOTHER | 2 |
| NO | 0 |
| PLEASE | 0 |
| THANKYOU | 13 |
| WATER | 0 |
| YES | 83 |

BOOK did not dominate the final test predictions. The model shifted to a
different collapse pattern, with YES accounting for 83 of 129 predictions.

## 9. Class-wise metrics

| Gloss | Support | Precision | Recall | F1 | Predicted count |
|---|---:|---:|---:|---:|---:|
| BOOK | 12 | 0.2500 | 0.0833 | 0.1250 | 4 |
| EAT1 | 17 | 0.2222 | 0.2353 | 0.2286 | 18 |
| HELLO | 14 | 0.0000 | 0.0000 | 0.0000 | 2 |
| HELP | 13 | 0.1429 | 0.0769 | 0.1000 | 7 |
| MOTHER | 15 | 0.0000 | 0.0000 | 0.0000 | 2 |
| NO | 11 | 0.0000 | 0.0000 | 0.0000 | 0 |
| PLEASE | 13 | 0.0000 | 0.0000 | 0.0000 | 0 |
| THANKYOU | 12 | 0.2308 | 0.2500 | 0.2400 | 13 |
| WATER | 12 | 0.0000 | 0.0000 | 0.0000 | 0 |
| YES | 10 | 0.0964 | 0.8000 | 0.1720 | 83 |

## 10. Training diagnostics

1. **Did training loss move substantially below `ln(10) ≈ 2.3026`?**  
   Yes. It reached **1.226280** at epoch 23.
2. **Did training accuracy escape the approximately 10% random region?**  
   Yes. It reached **58.39%**.
3. **Did validation accuracy escape the approximately 10% random region?**  
   Partially. It reached **17.95%**, but remained low and unstable.
4. **Did prediction diversity improve?**  
   Yes. Validation diversity rose from two classes at epoch 1 to nine classes
   at epoch 23.
5. **Is BOOK still dominating predictions?**  
   No. BOOK was only four test predictions.
6. **How many unique classes were predicted?**  
   Seven of ten on the test split.
7. **Is there train/validation divergence?**  
   Yes. Training accuracy climbed to 58.39% while final validation accuracy was
   12.82% and test accuracy was 13.18%. Validation loss also rose as training
   loss fell.
8. **Is there class collapse?**  
   There is severe partial collapse, not single-class collapse. YES made up
   64.34% of test predictions, while NO, PLEASE, and WATER were never predicted.
9. **Are there NaN/Inf values?**  
   No. The run summary reported no NaN/Inf loss values and the reloaded test
   logits were finite.
10. **Does the checkpoint reload correctly with the native-10 mapping?**  
    Yes at the payload/model level: the mapping matches the manifest and the
    state dict loads strictly. However, the reloaded config path fields expose
    an infrastructure gap documented below.

Overall, this is **Case B with clear overfitting/collapse symptoms**: the
classifier learned the training set substantially better than the signer-
independent validation/test data. Therefore, the earlier failure was not
explained only by stopping at epoch 6.

## 11. Infrastructure issues encountered

### Output isolation

The native-10 wrapper was used instead of the generic
`src/asl_citizen/train.py` entry point because the generic entry point imports
hardcoded `asl_citizen_100` output constants. The wrapper drove the same
training loop with `native_10_config()`, and all written experiment artifacts
resolved under `outputs/asl_citizen_native_10/`.

### Checkpoint mapping-path round trip

The checkpoint payload correctly stores:

- `class_to_idx`
- `idx_to_class`
- `num_classes`

But `checkpoint.py` does not currently serialize
`class_to_idx_path` and `idx_to_class_path` in `full_config`. Consequently, a
reloaded config defaults those path fields to the 100-class configuration even
though the checkpoint payload itself contains the correct native-10 mapping.

The existing native-10 evaluation wrapper explicitly restores the native-10
mapping paths before constructing the dataloader. This kept the experiment
correct and isolated without changing the shared checkpoint infrastructure or
running another training experiment. No existing Alembic migrations, models,
frontend systems, A–Z pipeline, word spelling system, database data, or
ASL Citizen 100-class experiment were modified.

## 12. Diagnosis

The controlled retry disproved the narrow hypothesis that the previous result
was only caused by patience 5. Training continued to epoch 23 and showed real
optimization:

- loss fell from `2.666678` to `1.226280`;
- training accuracy rose from `8.72%` to `58.39%`;
- validation accuracy briefly improved to `17.95%`;
- validation prediction diversity increased substantially.

However, generalization remained poor. The large train/validation gap, rising
validation loss, low test accuracy, and YES-heavy test distribution indicate
overfitting and representation/generalization problems in this frozen-backbone
configuration. The result is not a complete one-class BOOK collapse, but it is
still a serious partial class-collapse pattern.

## 13. Recommended NEXT experiment

Do not launch it automatically. The next experiment should be separately
approved and should change only one controlled factor. The most informative
next run is a **layer4-only fine-tuning experiment** using the same native-10
manifest, signer split, seed, transforms, temporal sampling, batch size, and
training duration, while preserving the same evaluation report. This directly
tests whether the frozen ImageNet representation is insufficient for the
native-sign domain.

If layer4 fine-tuning is not approved, the alternative should be a focused
preprocessing/temporal-sampling diagnostic rather than another blind frozen-
backbone retry. No native-10 result should be integrated into production yet.
