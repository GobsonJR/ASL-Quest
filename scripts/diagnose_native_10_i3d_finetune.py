"""Phase 8 Task 12/13/14: representation analysis for the partially fine-tuned I3D
model's test features, repeating the exact Phase 7 methodology for direct comparison.

Operates on the cached test features written by
scripts/evaluate_native_10_i3d_finetune.py (no model forward pass here)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FEATURES_PATH = ROOT / "outputs" / "asl_citizen_native_10_i3d" / "finetune" / "features" / "test_features.npz"
OUT = ROOT / "outputs" / "asl_citizen_native_10_i3d" / "finetune" / "diagnostics"

PHASE_4B_CLASS_SIL, PHASE_4B_SIGNER_SIL = -0.0725, 0.205
PHASE_4C_CLASS_SIL, PHASE_4C_SIGNER_SIL = -0.0769, 0.094
PHASE_6_CLASS_SIL, PHASE_6_SIGNER_SIL = -0.0723, 0.2167
PHASE_7_CLASS_SIL, PHASE_7_SIGNER_SIL = 0.22, -0.0612


def dump(name: str, payload: Any) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"  wrote {path.relative_to(ROOT)}")
    return path


def main() -> None:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"{FEATURES_PATH} not found — run scripts/evaluate_native_10_i3d_finetune.py first")
    OUT.mkdir(parents=True, exist_ok=True)

    data = np.load(FEATURES_PATH, allow_pickle=False)
    feats = data["features"].astype(np.float32)
    labels = data["labels"].astype(np.int64)
    participants = data["participants"].tolist()

    num_classes = int(labels.max()) + 1
    signer_ids = sorted(set(participants))
    signer_to_idx = {s: i for i, s in enumerate(signer_ids)}
    signer_labels = np.array([signer_to_idx[p] for p in participants])

    sil_class = float(silhouette_score(feats, labels)) if num_classes > 1 else None
    sil_signer = float(silhouette_score(feats, signer_labels)) if len(signer_ids) > 1 else None

    centroids = np.stack(
        [feats[labels == c].mean(axis=0) if np.any(labels == c) else np.zeros(feats.shape[1]) for c in range(num_classes)]
    )
    dist_to_centroid = np.linalg.norm(feats[:, None, :] - centroids[None, :, :], axis=-1)
    nearest_centroid_pred = dist_to_centroid.argmin(axis=1)
    nearest_centroid_acc = float((nearest_centroid_pred == labels).mean())

    pca = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(feats)
    explained = [round(float(v), 4) for v in pca.explained_variance_ratio_]

    for color_by, color_labels, cmap_name in (("class", labels, "tab10"), ("signer", signer_labels, "tab20")):
        fig, ax = plt.subplots(figsize=(6, 5))
        scatter = ax.scatter(proj[:, 0], proj[:, 1], c=color_labels, cmap=cmap_name, s=24)
        ax.set_title(
            f"I3D partial fine-tune — test features colored by {color_by}\nPCA explained var={explained}",
            fontsize=9,
        )
        legend = ax.legend(*scatter.legend_elements(num=min(10, len(set(color_labels)))), fontsize=6, loc="best", ncol=2)
        ax.add_artist(legend)
        fig.tight_layout()
        plot_path = OUT / f"i3d_finetune_pca_by_{color_by}.png"
        fig.savefig(plot_path, dpi=110)
        plt.close(fig)
        print(f"  wrote {plot_path.relative_to(ROOT)}")

    feature_report = {
        "i3d_finetune": {
            "num_test_samples": int(len(feats)),
            "feature_dim": int(feats.shape[1]),
            "nearest_centroid_accuracy": round(nearest_centroid_acc, 4),
            "silhouette_score_by_true_class": round(sil_class, 4) if sil_class is not None else None,
            "silhouette_score_by_signer": round(sil_signer, 4) if sil_signer is not None else None,
            "pca_explained_variance_ratio": explained,
            "num_unique_test_signers": len(signer_ids),
        }
    }
    dump("feature_analysis.json", feature_report)

    comparison = {
        "phase_4b_frozen_resnet": {"class_silhouette": PHASE_4B_CLASS_SIL, "signer_silhouette": PHASE_4B_SIGNER_SIL},
        "phase_4c_layer4_resnet": {"class_silhouette": PHASE_4C_CLASS_SIL, "signer_silhouette": PHASE_4C_SIGNER_SIL},
        "phase_6_appearance_aug": {"class_silhouette": PHASE_6_CLASS_SIL, "signer_silhouette": PHASE_6_SIGNER_SIL},
        "phase_7_i3d_frozen": {"class_silhouette": PHASE_7_CLASS_SIL, "signer_silhouette": PHASE_7_SIGNER_SIL},
        "phase_8_i3d_finetune": {
            "class_silhouette": round(sil_class, 4) if sil_class is not None else None,
            "signer_silhouette": round(sil_signer, 4) if sil_signer is not None else None,
        },
        "class_silhouette_improved_vs_frozen_i3d": (sil_class is not None) and (sil_class > PHASE_7_CLASS_SIL),
        "signer_silhouette_decreased_vs_frozen_i3d": (sil_signer is not None) and (sil_signer < PHASE_7_SIGNER_SIL),
    }
    dump("silhouette_comparison.json", comparison)
    print("\nSilhouette comparison:")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
