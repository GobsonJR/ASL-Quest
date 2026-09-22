from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.data.dataset import build_dataloaders
from src.models.resnet18 import build_resnet18
from src.utils.checkpoint import build_checkpoint, load_checkpoint, save_checkpoint
from src.utils.device import print_device_info


def main() -> None:
    device = print_device_info()
    train_loader, val_loader, test_loader, class_names, splits = build_dataloaders(
        data_dir=ROOT / "dataset" / "asl_alphabet_train",
        batch_size=8,
        num_workers=0,
        max_per_class=4,
    )
    model = build_resnet18(num_classes=len(class_names), pretrained=False).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    images, labels = next(iter(train_loader))
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    loss = criterion(outputs, labels)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    print(f"Smoke batch ok. classes={len(class_names)} loss={loss.item():.4f}")
    print(f"split sizes train/val/test={len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")
    model.eval()
    with torch.no_grad():
        val_images, val_labels = next(iter(val_loader))
        val_logits = model(val_images.to(device))
        val_loss = criterion(val_logits, val_labels.to(device))
        test_images, _ = next(iter(test_loader))
        _ = model(test_images.to(device))
    out = ROOT / "outputs" / "smoke_test" / "smoke_checkpoint.pth"
    save_checkpoint(
        build_checkpoint(
            model,
            class_names,
            {
                "smoke_test": True,
                "train_loss": float(loss.item()),
                "val_loss": float(val_loss.item()),
                "split_sizes": {name: len(indices) for name, indices in splits.items()},
            },
        ),
        out,
    )
    payload = load_checkpoint(out)
    assert payload["class_names"] == class_names
    print(f"Evaluation pass ok. val_loss={val_loss.item():.4f}")
    print(f"Checkpoint save/load ok: {out}")
    print("DataLoader iteration ok.")


if __name__ == "__main__":
    main()
