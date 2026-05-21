from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from .base import BaseEncoder


MEDICAL_MODEL_NAMES = {
    "medical_resnet18",
    "medical_resnet34",
    "medical_resnet50",
    "medical_densenet121",
    "medical_efficientnet_b0",
}


def load_medical_array(image_path: str, preserve_rgb: bool = False) -> np.ndarray:
    path = Path(image_path)
    suffix = path.suffix.lower()

    if suffix == ".dcm":
        try:
            import pydicom
        except ImportError as exc:
            raise ImportError(
                "DICOM support requires pydicom. Install it with: pip install pydicom"
            ) from exc

        ds = pydicom.dcmread(str(path))
        arr = ds.pixel_array.astype(np.float32)

        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept

        return arr.astype(np.float32)

    image = Image.open(str(path))

    # Untuk dataset RGB seperti BloodMNIST,
    # jangan ubah gambar menjadi grayscale.
    if preserve_rgb and image.mode in {"RGB", "RGBA", "P", "CMYK"}:
        return np.asarray(image.convert("RGB")).astype(np.float32)

    image = image.convert("L")
    return np.asarray(image).astype(np.float32)


def window_ct(arr: np.ndarray, center: float, width: float) -> np.ndarray:
    low = center - width / 2.0
    high = center + width / 2.0
    arr = np.clip(arr, low, high)
    arr = (arr - low) / (high - low + 1e-8)
    return arr


def normalize_medical_array(arr: np.ndarray, mode: str = "percentile") -> np.ndarray:
    # Jika input RGB, tidak perlu dilakukan CT windowing atau percentile grayscale.
    # Cukup pastikan nilainya berada pada rentang 0-255 uint8.
    if arr.ndim == 3:
        arr = arr.astype(np.float32)

        if arr.max() > 1.0:
            arr = arr / 255.0

        arr = np.clip(arr, 0.0, 1.0)
        return (arr * 255.0).astype(np.uint8)

    mode = mode.lower() if isinstance(mode, str) else mode

    if mode == "ct_lung":
        arr = window_ct(arr, center=-600, width=1500)
    elif mode == "ct_soft":
        arr = window_ct(arr, center=40, width=400)
    elif mode == "ct_brain":
        arr = window_ct(arr, center=40, width=80)
    elif mode == "percentile":
        p1, p99 = np.percentile(arr, [1, 99])
        arr = np.clip(arr, p1, p99)
        arr = (arr - p1) / (p99 - p1 + 1e-8)
    elif mode == "minmax":
        mn, mx = float(np.min(arr)), float(np.max(arr))
        arr = (arr - mn) / (mx - mn + 1e-8)
    elif mode in ("none", None):
        mn, mx = float(np.min(arr)), float(np.max(arr))
        arr = (arr - mn) / (mx - mn + 1e-8)
    else:
        raise ValueError(
            "intensity_mode must be one of: "
            "'ct_lung', 'ct_soft', 'ct_brain', 'percentile', 'minmax', 'none'."
        )

    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def array_to_pil_rgb(arr: np.ndarray) -> Image.Image:
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L").convert("RGB")

    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
            return Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")

        return Image.fromarray(arr.astype(np.uint8)).convert("RGB")

    raise ValueError(f"Unsupported medical image shape: {arr.shape}")


def build_tensor_transform(
    image_size: int = 224,
    normalize: str = "imagenet",
) -> transforms.Compose:
    ops = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]

    if normalize == "imagenet":
        ops.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )
    elif normalize == "half":
        ops.append(
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            )
        )
    elif normalize in ("none", None):
        pass
    else:
        raise ValueError("normalize must be one of: 'imagenet', 'half', 'none'.")

    return transforms.Compose(ops)


class MedicalPreprocessor:
    def __init__(
        self,
        image_size: int = 224,
        intensity_mode: str = "percentile",
        normalize: str = "imagenet",
        preserve_rgb: bool = False,
    ) -> None:
        self.image_size = int(image_size)
        self.intensity_mode = intensity_mode
        self.normalize = normalize
        self.preserve_rgb = preserve_rgb

        self.transform = build_tensor_transform(
            image_size=self.image_size,
            normalize=self.normalize,
        )

    def load_pil(self, image_path: str) -> Image.Image:
        arr = load_medical_array(
            image_path=image_path,
            preserve_rgb=self.preserve_rgb,
        )
        arr = normalize_medical_array(arr, mode=self.intensity_mode)
        return array_to_pil_rgb(arr)

    def __call__(self, image_path: str) -> torch.Tensor:
        image = self.load_pil(image_path)
        return self.transform(image)


def build_medical_model(
    name: str,
    pretrained: bool = True,
    num_classes: Optional[int] = None,
    feature_mode: bool = True,
) -> Tuple[nn.Module, int]:
    name = name.lower()

    if name == "medical_resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        dim = model.fc.in_features
        model.fc = nn.Identity() if feature_mode else nn.Linear(dim, int(num_classes))
        return model, dim

    if name == "medical_resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
        dim = model.fc.in_features
        model.fc = nn.Identity() if feature_mode else nn.Linear(dim, int(num_classes))
        return model, dim

    if name == "medical_resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        dim = model.fc.in_features
        model.fc = nn.Identity() if feature_mode else nn.Linear(dim, int(num_classes))
        return model, dim

    if name == "medical_densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        dim = model.classifier.in_features
        model.classifier = nn.Identity() if feature_mode else nn.Linear(dim, int(num_classes))
        return model, dim

    if name == "medical_efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        dim = model.classifier[1].in_features
        model.classifier = nn.Identity() if feature_mode else nn.Linear(dim, int(num_classes))
        return model, dim

    raise ValueError(f"Unsupported medical encoder: {name}")


def clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}

    for key, value in state_dict.items():
        key = key.replace("module.", "")
        key = key.replace("model.", "")
        cleaned[key] = value

    return cleaned


def extract_checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if "backbone_state_dict" in checkpoint:
            return checkpoint["backbone_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]

    return checkpoint


class MedicalImageEncoder(BaseEncoder):
    def __init__(
        self,
        name: str = "medical_resnet50",
        pretrained: bool = True,
        device: Optional[str] = None,
        image_size: int = 224,
        intensity_mode: str = "percentile",
        normalize: str = "imagenet",
        checkpoint_path: Optional[str] = None,
        preserve_rgb: bool = False,
        **kwargs,
    ) -> None:
        self.name = name.lower()
        self.pretrained = pretrained
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = int(image_size)
        self.intensity_mode = intensity_mode
        self.normalize = normalize
        self.checkpoint_path = checkpoint_path
        self.preserve_rgb = preserve_rgb

        self.model, self.embedding_dim = build_medical_model(
            name=self.name,
            pretrained=self.pretrained,
            feature_mode=True,
        )

        self.model.to(self.device)
        self.model.eval()

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

        for param in self.model.parameters():
            param.requires_grad = False

        self.preprocessor = MedicalPreprocessor(
            image_size=self.image_size,
            intensity_mode=self.intensity_mode,
            normalize=self.normalize,
            preserve_rgb=self.preserve_rgb,
        )

    def load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = extract_checkpoint_state_dict(checkpoint)
        state_dict = clean_state_dict(state_dict)

        self.model.load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def encode_path(self, image_path: str) -> np.ndarray:
        tensor = self.preprocessor(image_path)
        tensor = tensor.unsqueeze(0).to(self.device)

        embedding = self.model(tensor)

        if isinstance(embedding, tuple):
            embedding = embedding[0]

        embedding = embedding.squeeze(0)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)

        return embedding.detach().cpu().numpy().astype(np.float32)