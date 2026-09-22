"""Phase 6 representation analysis: repeats the Phase 5 feature/signer diagnostic
(scripts/diagnose_native_10_phase5.py's diagnostic_5_and_6) against the new
appearance-augmentation best checkpoint, using the exact same
run_inference_with_features() procedure (imported, not reimplemented) so the numbers
are directly comparable to the Phase 4B (frozen) and Phase 4C (layer4) figures already
on record:

    Phase 4B frozen : class silhouette = -0.0725, signer silhouette = 0.205
    Phase 4C layer4 : class silhouette = -0.0769, signer silhouette = 0.094

Read-only on checkpoints/manifests/dataset videos. Writes only under
outputs/asl_citizen_native_10/experiments/appearance_aug/diagnostics/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_native_10_phase5 import run_inference_with_features
from collections import Counter

APPEARANCE_AUG_BEST = (
    ROOT / "outputs" / "asl_citizen_native_10" / "experiments" / "appearance_aug" / "checkpoints"
    / "asl_citizen_native_10_appearance_aug_resnet18_gru_best.pth"
)
OUT = ROOT / "outputs" / "asl_citizen_native_10" / "experiments" / "appearance_aug" / "diagnostics"

PHASE_4B_SIGNER_SILHOUETTE = 0.205
PHASE_4B_CLASS_SILHOUETTE = -0.0725
PHASE_4C_SIGNER_SILHOUETTE = 0.094
PHASE_4C_CLASS_SILHOUETTE = -0.0769


def dump(name: str, payload: Any) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"  wrote {path.relative_to(ROOT)}")
    return path


def main() -> None:
    if not APPEARANCE_AUG_BEST.exists():
        raise FileNotFoundError(f"Phase 6 best checkpoint not found: {APPEARANCE_AUG_BEST}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    OUT.mkdir(parents=True, exist_ok=True)

    result = run_inference_with_features(APPEARANCE_AUG_BEST, "test", device)
    feats = result["features"]
    labels = result["labels"]
    preds = result["preds"]
    participants = result["participants"]
    idx_to_class = result["idx_to_class"]
    num_classes = len(idx_to_class)

    centroids = np.stack(
        [feats[labels == c].mean(axis=0) if np.any(labels == c) else np.zeros(feats.shape[1]) for c in range(num_classes)]
    )
    intra = {}
    for c in range(num_classes):
        members = feats[labels == c]
        if len(members) > 1:
            d = np.linalg.norm(members[:, None, :] - members[None, :, :], axis=-1)
            intra[idx_to_class[str(c)]] = round(float(d[np.triu_indices(len(members), k=1)].mean()), 4)
        else:
            intra[idx_to_class[str(c)]] = None
    inter_matrix = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=-1)
    inter_mean = float(inter_matrix[np.triu_indices(num_classes, k=1)].mean())

    dist_to_centroid = np.linalg.norm(feats[:, None, :] - centroids[None, :, :], axis=-1)
    nearest_centroid_pred = dist_to_centroid.argmin(axis=1)
    nearest_centroid_acc = float((nearest_centroid_pred == labels).mean())
    classifier_acc = float((preds == labels).mean())

    signer_ids = sorted(set(participants))
    signer_to_idx = {s: i for i, s in enumerate(signer_ids)}
    signer_labels = np.array([signer_to_idx[p] for p in participants])
    sil_class = float(silhouette_score(feats, labels)) if num_classes > 1 else None
    sil_signer = float(silhouette_score(feats, signer_labels)) if len(signer_ids) > 1 else None

    pca = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(feats)
    explained = [round(float(v), 4) for v in pca.explained_variance_ratio_]

    for color_by, color_labels, cmap_name in (("class", labels, "tab10"), ("signer", signer_labels, "tab20")):
        fig, ax = plt.subplots(figsize=(6, 5))
        scatter = ax.scatter(proj[:, 0], proj[:, 1], c=color_labels, cmap=cmap_name, s=24)
        ax.set_title(
            f"appearance_aug checkpoint — test features colored by {color_by}\nPCA explained var={explained}",
            fontsize=9,
        )
        legend = ax.legend(*scatter.legend_elements(num=min(10, len(set(color_labels)))), fontsize=6, loc="best", ncol=2)
        ax.add_artist(legend)
        fig.tight_layout()
        plot_path = OUT / f"appearance_aug_pca_by_{color_by}.png"
        fig.savefig(plot_path, dpi=110)
        plt.close(fig)
        print(f"  wrote {plot_path.relative_to(ROOT)}")

    feature_report = {
        "appearance_aug": {
            "num_test_samples": int(len(feats)),
            "feature_dim": int(feats.shape[1]),
            "intra_class_mean_distance": intra,
            "inter_class_centroid_mean_distance": round(inter_mean, 4),
            "nearest_centroid_accuracy": round(nearest_centroid_acc, 4),
            "classifier_top1_accuracy_cross_check": round(classifier_acc, 4),
            "silhouette_score_by_true_class": round(sil_class, 4) if sil_class is not None else None,
            "silhouette_score_by_signer": round(sil_signer, 4) if sil_signer is not None else None,
            "pca_explained_variance_ratio": explained,
            "num_unique_test_signers": len(signer_ids),
        }
    }
    dump("feature_analysis.json", feature_report)

    per_signer: dict[str, Any] = {}
    for signer in signer_ids:
        mask = [p == signer for p in participants]
        idxs = [i for i, m in enumerate(mask) if m]
        s_labels = labels[idxs]
        s_preds = preds[idxs]
        pred_dist = Counter(idx_to_class[str(p)] for p in s_preds)
        per_signer[signer] = {
            "total": len(idxs),
            "correct": int((s_labels == s_preds).sum()),
            "accuracy": round(float((s_labels == s_preds).mean()), 4) if idxs else None,
            "predicted_class_distribution": dict(pred_dist),
        }
    dump("signer_analysis_predictions.json", {"appearance_aug": per_signer})

    comparison = {
        "phase_4b_frozen": {
            "class_silhouette": PHASE_4B_CLASS_SILHOUETTE,
            "signer_silhouette": PHASE_4B_SIGNER_SILHOUETTE,
        },
        "phase_4c_layer4": {
            "class_silhouette": PHASE_4C_CLASS_SILHOUETTE,
            "signer_silhouette": PHASE_4C_SIGNER_SILHOUETTE,
        },
        "phase_6_appearance_aug": {
            "class_silhouette": round(sil_class, 4) if sil_class is not None else None,
            "signer_silhouette": round(sil_signer, 4) if sil_signer is not None else None,
        },
        "class_silhouette_improved_vs_4b": (sil_class is not None) and (sil_class > PHASE_4B_CLASS_SILHOUETTE),
        "signer_silhouette_decreased_vs_4b": (sil_signer is not None) and (sil_signer < PHASE_4B_SIGNER_SILHOUETTE),
    }
    dump("silhouette_comparison.json", comparison)
    print("\nSilhouette comparison:")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
