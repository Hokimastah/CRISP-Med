from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .datasets import FolderMedicalDataset
from .encoders.medical import build_medical_model


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_indices(n: int, val_split: float, seed: int = 42):
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    if val_split <= 0:
        return indices, []

    val_size = int(round(n * val_split))

    if n > 1:
        val_size = max(1, min(val_size, n - 1))
    else:
        val_size = 0

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    return train_indices, val_indices


def compute_class_weights(dataset: FolderMedicalDataset, num_classes: int) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.float32)

    for _, label in dataset.samples:
        counts[int(label)] += 1

    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)

    return torch.tensor(weights, dtype=torch.float32)


def extract_backbone_state_dict(model: nn.Module, encoder: str) -> Dict[str, torch.Tensor]:
    state = model.state_dict()
    encoder = encoder.lower()

    if encoder.startswith("medical_resnet"):
        return {k: v for k, v in state.items() if not k.startswith("fc.")}

    if encoder in {"medical_densenet121", "medical_efficientnet_b0"}:
        return {k: v for k, v in state.items() if not k.startswith("classifier.")}

    return dict(state)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total = 0
    correct = 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * images.size(0)
        total += int(labels.numel())
        correct += int((logits.argmax(dim=1) == labels).sum().item())

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
    }


@torch.no_grad()
def evaluate_classifier(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total = 0
    correct = 0

    for images, labels in tqdm(loader, desc="Val", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += float(loss.item()) * images.size(0)
        total += int(labels.numel())
        correct += int((logits.argmax(dim=1) == labels).sum().item())

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
    }


def train_medical_backbone(
    data_dir: str,
    encoder: str = "medical_resnet50",
    output: str = "checkpoints/medical_backbone.pth",
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    image_size: int = 224,
    intensity_mode: str = "percentile",
    normalize: str = "imagenet",
    val_split: float = 0.2,
    pretrained: bool = True,
    device: Optional[str] = None,
    num_workers: int = 0,
    seed: int = 42,
    class_weighted_loss: bool = False,
):
    set_seed(seed)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    dataset = FolderMedicalDataset(
        root=data_dir,
        image_size=image_size,
        intensity_mode=intensity_mode,
        normalize=normalize,
    )

    num_classes = len(dataset.class_to_idx)

    if num_classes < 2:
        raise ValueError("Training requires at least two class folders.")

    train_idx, val_idx = split_indices(
        n=len(dataset),
        val_split=val_split,
        seed=seed,
    )

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx) if val_idx else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    val_loader = None

    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device == "cuda"),
        )

    model, embedding_dim = build_medical_model(
        name=encoder,
        pretrained=pretrained,
        num_classes=num_classes,
        feature_mode=False,
    )

    model.to(device)

    if class_weighted_loss:
        weights = compute_class_weights(dataset, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_metric = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        if val_loader is not None:
            val_metrics = evaluate_classifier(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
            )
            selection_metric = val_metrics["accuracy"]
        else:
            val_metrics = {"loss": None, "accuracy": None}
            selection_metric = train_metrics["accuracy"]

        if selection_metric > best_metric:
            best_metric = selection_metric
            best_state = copy.deepcopy(model.state_dict())

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }

        history.append(record)

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"val_acc={val_metrics['accuracy'] if val_metrics['accuracy'] is not None else 'NA'}"
        )

    model.load_state_dict(best_state)

    checkpoint = {
        "format": "crisp-med-checkpoint-v1",
        "encoder": encoder,
        "num_classes": num_classes,
        "embedding_dim": embedding_dim,
        "class_to_idx": dataset.class_to_idx,
        "idx_to_class": dataset.idx_to_class,
        "preprocessing": {
            "image_size": image_size,
            "intensity_mode": intensity_mode,
            "normalize": normalize,
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "val_split": val_split,
            "pretrained": pretrained,
            "best_metric": best_metric,
            "history": history,
        },
        "state_dict": model.state_dict(),
        "backbone_state_dict": extract_backbone_state_dict(model, encoder),
    }

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, output_path)

    print(f"Saved checkpoint: {output_path}")

    return checkpoint