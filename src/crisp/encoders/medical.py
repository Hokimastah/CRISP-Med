from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

from .base import BaseEncoder, l2_normalize


_MEDICAL_BACKBONES = {
    "medical_resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT, 512),
    "medical_resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT, 512),
    "medical_resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, 2048),
    "medical_densenet121": (models.densenet121, models.DenseNet121_Weights.DEFAULT, 1024),
    "medical_efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT, 1280),
}


def _read_dicom(path: str) -> np.ndarray:
    try:
        import pydicom
    except ImportError as exc:
        raise ImportError(
            "DICOM support requires pydicom. Install with: pip install -e '.[medical]'"
        ) from exc

    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return arr * slope + intercept


def _read_image_as_array(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix.lower() == ".dcm":
        return _read_dicom(path)
    image = Image.open(path)
    if image.mode not in ("L", "I;16", "F"):
        image = image.convert("L")
    return np.asarray(image, dtype=np.float32)


def _window_ct(arr: np.ndarray, center: float, width: float) -> np.ndarray:
    low = center - width / 2.0
    high = center + width / 2.0
    arr = np.clip(arr, low, high)
    return (arr - low) / max(high - low, 1e-6)


def preprocess_medical_array(arr: np.ndarray, mode: str = "percentile") -> np.ndarray:
    """
    Convert grayscale medical image / DICOM pixel data into [0, 1].

    Available modes:
    - percentile: robust 1st-99th percentile scaling
    - minmax: min-max scaling
    - ct_lung: CT lung window, center=-600, width=1500
    - ct_soft: CT soft tissue window, center=40, width=400
    - ct_brain: CT brain window, center=40, width=80
    """
    arr = np.asarray(arr, dtype=np.float32)
    mode = mode.lower()

    if mode == "ct_lung":
        out = _window_ct(arr, center=-600.0, width=1500.0)
    elif mode == "ct_soft":
        out = _window_ct(arr, center=40.0, width=400.0)
    elif mode == "ct_brain":
        out = _window_ct(arr, center=40.0, width=80.0)
    elif mode == "minmax":
        mn, mx = float(np.min(arr)), float(np.max(arr))
        out = (arr - mn) / max(mx - mn, 1e-6)
    elif mode == "percentile":
        lo, hi = np.percentile(arr, [1, 99])
        arr = np.clip(arr, lo, hi)
        out = (arr - lo) / max(hi - lo, 1e-6)
    else:
        raise ValueError(
            "intensity_mode must be one of: percentile, minmax, ct_lung, ct_soft, ct_brain"
        )

    return np.clip(out, 0.0, 1.0).astype(np.float32)


class MedicalImageEncoder(BaseEncoder):
    """
    Frozen 2D medical image encoder.

    This encoder is intended for 2D medical images or selected 2D slices from
    CT/MRI/DICOM volumes. It is a baseline encoder. For serious experiments,
    load a domain-specific checkpoint trained on similar modality/anatomy.
    """

    def __init__(
        self,
        name: str = "medical_resnet50",
        pretrained: bool = True,
        device: Optional[str] = None,
        image_size: int = 224,
        intensity_mode: str = "percentile",
        checkpoint_path: Optional[str] = None,
    ) -> None:
        name = name.lower()
        if name not in _MEDICAL_BACKBONES:
            raise ValueError(f"Unsupported medical encoder: {name}")

        self.name = name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.intensity_mode = intensity_mode

        builder, weights, output_dim = _MEDICAL_BACKBONES[name]
        self.output_dim = output_dim
        model = builder(weights=weights if pretrained else None)

        if name.startswith("medical_resnet"):
            model.fc = torch.nn.Identity()
        elif name == "medical_densenet121":
            model.classifier = torch.nn.Identity()
        elif name == "medical_efficientnet_b0":
            model.classifier = torch.nn.Identity()

        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            cleaned = {k.replace("module.", ""): v for k, v in state.items()}
            missing, unexpected = model.load_state_dict(cleaned, strict=False)
            if unexpected:
                print(f"Warning: unexpected checkpoint keys: {unexpected[:5]}")
            if missing:
                print(f"Warning: missing checkpoint keys: {missing[:5]}")

        model.eval().to(self.device)
        self.model = model
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def encode_path(self, image_path: str) -> np.ndarray:
        arr = _read_image_as_array(image_path)
        arr = preprocess_medical_array(arr, mode=self.intensity_mode)
        arr_u8 = (arr * 255.0).astype(np.uint8)
        image = Image.fromarray(arr_u8, mode="L").convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(tensor).detach().cpu().numpy()[0]
        return l2_normalize(embedding)
