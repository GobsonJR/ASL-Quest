import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim


# ============================================================
# 1. SETTINGS
# ============================================================

DATASET_PATH = r"D:\SIGN LANGUAGE\dataset\asl_alphabet_train"

IMAGE_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001

MODEL_PATH = "asl_resnet18_best.pth"


# ============================================================
# 2. MAIN FUNCTION
# ============================================================

def main():

    # ========================================================
    # 3. GPU SETUP
    # ========================================================

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("DEVICE:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA:", torch.version.cuda)

        # Optional: show GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        print(
            "GPU Memory:",
            round(gpu_memory / (1024 ** 3), 2),
            "GB"
        )

    else:
        print("WARNING: CUDA GPU not detected.")

    print("=" * 60)


    # ========================================================
    # 4. IMAGE TRANSFORMS
    # ========================================================

    train_transform = transforms.Compose([

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.RandomRotation(10),

        transforms.RandomHorizontalFlip(),

        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


    val_transform = transforms.Compose([

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


    # ========================================================
    # 5. LOAD DATASET
    # ========================================================

    print("\nLoading dataset...")

    full_dataset = datasets.ImageFolder(
        root=DATASET_PATH,
        transform=train_transform
    )

    print("Total images:", len(full_dataset))

    print(
        "Classes:",
        full_dataset.classes
    )

    print(
        "Number of classes:",
        len(full_dataset.classes)
    )


    # ========================================================
    # 6. CHECK DATASET
    # ========================================================

    if len(full_dataset.classes) != 26:

        raise ValueError(
            f"Expected 26 classes, "
            f"but found {len(full_dataset.classes)}"
        )


    if len(full_dataset) != 78000:

        print(
            "\nWARNING:"
            f"Expected approximately 78000 images, "
            f"but found {len(full_dataset)}."
        )


    # ========================================================
    # 7. TRAIN / VALIDATION SPLIT
    # ========================================================

    train_size = int(
        0.8 * len(full_dataset)
    )

    val_size = (
        len(full_dataset) - train_size
    )

    generator = torch.Generator().manual_seed(42)

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator
    )


    # IMPORTANT:
    # Both subsets reference the same ImageFolder object.
    # We create separate datasets below so that the
    # validation set does NOT use training augmentation.

    train_base = datasets.ImageFolder(
        root=DATASET_PATH,
        transform=train_transform
    )

    val_base = datasets.ImageFolder(
        root=DATASET_PATH,
        transform=val_transform
    )


    train_dataset, val_dataset = random_split(
        train_base,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )


    # Reuse exactly the same indices for validation

    _, val_indices = random_split(
        range(len(val_base)),
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    val_dataset = torch.utils.data.Subset(
        val_base,
        val_indices
    )


    print(
        "\nTraining images:",
        len(train_dataset)
    )

    print(
        "Validation images:",
        len(val_dataset)
    )


    # ========================================================
    # 8. DATA LOADERS
    # ========================================================

    print("\nCreating DataLoaders...")

    train_loader = DataLoader(
        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        # IMPORTANT FOR WINDOWS
        num_workers=0,

        pin_memory=True
    )


    val_loader = DataLoader(
        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        # IMPORTANT FOR WINDOWS
        num_workers=0,

        pin_memory=True
    )


    # ========================================================
    # 9. LOAD PRETRAINED RESNET18
    # ========================================================

    print("\nLoading ResNet18...")

    weights = models.ResNet18_Weights.DEFAULT

    model = models.resnet18(
        weights=weights
    )


    # ========================================================
    # 10. REPLACE FINAL CLASSIFICATION LAYER
    # ========================================================

    num_features = model.fc.in_features

    model.fc = nn.Linear(
        num_features,
        26
    )


    model = model.to(device)


    # ========================================================
    # 11. LOSS FUNCTION
    # ========================================================

    criterion = nn.CrossEntropyLoss()


    # ========================================================
    # 12. OPTIMIZER
    # ========================================================

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )


    # ========================================================
    # 13. TRAINING VARIABLES
    # ========================================================

    best_accuracy = 0.0


    # ========================================================
    # 14. TRAINING LOOP
    # ========================================================

    print("\nStarting training...")
    print("=" * 60)


    for epoch in range(EPOCHS):

        # ====================================================
        # TRAINING
        # ====================================================

        model.train()

        running_loss = 0.0

        correct = 0

        total = 0


        for images, labels in train_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )


            # Clear gradients

            optimizer.zero_grad()


            # Forward pass

            outputs = model(images)


            # Calculate loss

            loss = criterion(
                outputs,
                labels
            )


            # Backpropagation

            loss.backward()


            # Update weights

            optimizer.step()


            # Statistics

            running_loss += loss.item()


            _, predicted = torch.max(
                outputs,
                1
            )


            total += labels.size(0)


            correct += (
                predicted == labels
            ).sum().item()


        train_accuracy = (
            100.0 * correct / total
        )


        train_loss = (
            running_loss /
            len(train_loader)
        )


        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        val_correct = 0

        val_total = 0

        val_loss = 0.0


        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(
                    device,
                    non_blocking=True
                )

                labels = labels.to(
                    device,
                    non_blocking=True
                )


                outputs = model(images)


                loss = criterion(
                    outputs,
                    labels
                )


                val_loss += loss.item()


                _, predicted = torch.max(
                    outputs,
                    1
                )


                val_total += labels.size(0)


                val_correct += (
                    predicted == labels
                ).sum().item()


        validation_accuracy = (
            100.0 *
            val_correct /
            val_total
        )


        validation_loss = (
            val_loss /
            len(val_loader)
        )


        # ====================================================
        # PRINT RESULTS
        # ====================================================

        print(
            f"\nEpoch [{epoch + 1}/{EPOCHS}]"
        )

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_accuracy:.2f}%"
        )

        print(
            f"Validation Loss: "
            f"{validation_loss:.4f}"
        )

        print(
            f"Validation Accuracy: "
            f"{validation_accuracy:.2f}%"
        )


        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if validation_accuracy > best_accuracy:

            best_accuracy = validation_accuracy


            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "classes":
                        full_dataset.classes,

                    "accuracy":
                        best_accuracy
                },

                MODEL_PATH
            )


            print(
                "\n✓ BEST MODEL SAVED"
            )

            print(
                f"Best Validation Accuracy: "
                f"{best_accuracy:.2f}%"
            )


    # ========================================================
    # 15. TRAINING COMPLETE
    # ========================================================

    print("\n")
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Best Validation Accuracy: "
        f"{best_accuracy:.2f}%"
    )

    print(
        f"Model saved to: "
        f"{MODEL_PATH}"
    )

    print("=" * 60)


# ============================================================
# 16. WINDOWS ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()

