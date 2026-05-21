from __future__ import annotations

from pathlib import Path
from typing import Optional

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


class MedicalImageEncoder(BaseEncoder):
    """
    Medical image encoder for CRISP-Med.

    Supported:
    - medical_resnet18
    - medical_resnet34
    - medical_resnet50
    - medical_densenet121
    - medical_efficientnet_b0

    Supports:
    - PNG/JPG/TIFF medical images
    - DICOM if pydicom is installed
    - CT windowing
    - grayscale-to-RGB conversion
    - configurable normalization
    """

    def __init__(
        self,
        name: str = "medical_resnet50",
        pretrained: bool = True,
        device: Optional[str] = None,
        image_size: int = 224,
        intensity_mode: str = "percentile",
        normalize: str = "imagenet",
        checkpoint_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.name = name.lower()
        self.pretrained = pretrained
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.intensity_mode = intensity_mode
        self.normalize = normalize
        self.checkpoint_path = checkpoint_path

        self.model, self.embedding_dim = self._build_model()
        self.model.to(self.device)
        self.model.eval()

        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

        self.transform = self._build_transform()

    def _build_model(self):
        if self.name == "medical_resnet18":
            weights = models.ResNet18_Weights.DEFAULT if self.pretrained else None
            model = models.resnet18(weights=weights)
            dim = model.fc.in_features
            model.fc = nn.Identity()
            return model, dim

        if self.name == "medical_resnet34":
            weights = models.ResNet34_Weights.DEFAULT if self.pretrained else None
            model = models.resnet34(weights=weights)
            dim = model.fc.in_features
            model.fc = nn.Identity()
            return model, dim

        if self.name == "medical_resnet50":
            weights = models.ResNet50_Weights.DEFAULT if self.pretrained else None
            model = models.resnet50(weights=weights)
            dim = model.fc.in_features
            model.fc = nn.Identity()
            return model, dim

        if self.name == "medical_densenet121":
            weights = models.DenseNet121_Weights.DEFAULT if self.pretrained else None
            model = models.densenet121(weights=weights)
            dim = model.classifier.in_features
            model.classifier = nn.Identity()
            return model, dim

        if self.name == "medical_efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if self.pretrained else None
            model = models.efficientnet_b0(weights=weights)
            dim = model.classifier[1].in_features
            model.classifier = nn.Identity()
            return model, dim

        raise ValueError(
            f"Unsupported medical encoder: {self.name}. "
            f"Supported: {sorted(MEDICAL_MODEL_NAMES)}"
        )

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        cleaned = {}
        for key, value in checkpoint.items():
            new_key = key.replace("module.", "")
            cleaned[new_key] = value

        self.model.load_state_dict(cleaned, strict=False)

    def _build_transform(self):
        transform_list = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ]

        if self.normalize == "imagenet":
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            )
        elif self.normalize == "half":
            transform_list.append(
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                )
            )
        elif self.normalize in ["none", None]:
            pass
        else:
            raise ValueError(
                "normalize must be one of: 'imagenet', 'half', 'none'."
            )

        return transforms.Compose(transform_list)

    def _load_image_array(self, image_path: str) -> np.ndarray:
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

            return arr

        image = Image.open(image_path).convert("L")
        return np.array(image).astype(np.float32)

    def _window_ct(self, arr: np.ndarray, center: float, width: float) -> np.ndarray:
        low = center - width / 2
        high = center + width / 2
        arr = np.clip(arr, low, high)
        arr = (arr - low) / (high - low + 1e-8)
        return arr

    def _normalize_medical_image(self, arr: np.ndarray) -> np.ndarray:
        mode = self.intensity_mode

        if mode == "ct_lung":
            arr = self._window_ct(arr, center=-600, width=1500)

        elif mode == "ct_soft":
            arr = self._window_ct(arr, center=40, width=400)

        elif mode == "ct_brain":
            arr = self._window_ct(arr, center=40, width=80)

        elif mode == "percentile":
            p1, p99 = np.percentile(arr, [1, 99])
            arr = np.clip(arr, p1, p99)
            arr = (arr - p1) / (p99 - p1 + 1e-8)

        elif mode == "minmax":
            arr_min, arr_max = arr.min(), arr.max()
            arr = (arr - arr_min) / (arr_max - arr_min + 1e-8)

        elif mode in ["none", None]:
            arr_min, arr_max = arr.min(), arr.max()
            arr = (arr - arr_min) / (arr_max - arr_min + 1e-8)

        else:
            raise ValueError(
                "intensity_mode must be one of: "
                "'ct_lung', 'ct_soft', 'ct_brain', 'percentile', 'minmax', 'none'."
            )

        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255).astype(np.uint8)

        return arr

    def _array_to_pil_rgb(self, arr: np.ndarray) -> Image.Image:
        if arr.ndim == 2:
            image = Image.fromarray(arr, mode="L").convert("RGB")
        elif arr.ndim == 3:
            image = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
        else:
            raise ValueError(f"Unsupported image shape: {arr.shape}")

        return image

    @torch.no_grad()
    def encode_path(self, image_path: str) -> np.ndarray:
        arr = self._load_image_array(image_path)
        arr = self._normalize_medical_image(arr)
        image = self._array_to_pil_rgb(arr)

        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0).to(self.device)

        embedding = self.model(tensor)

        if isinstance(embedding, tuple):
            embedding = embedding[0]

        embedding = embedding.squeeze(0)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)

        return embedding.detach().cpu().numpy().astype(np.float32)