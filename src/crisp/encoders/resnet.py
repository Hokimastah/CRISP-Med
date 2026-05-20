from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

from .base import BaseEncoder, l2_normalize


_RESNET_SPECS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT, 512),
    "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT, 512),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, 2048),
    "resnet101": (models.resnet101, models.ResNet101_Weights.DEFAULT, 2048),
    "resnet152": (models.resnet152, models.ResNet152_Weights.DEFAULT, 2048),
}


class ResNetEncoder(BaseEncoder):
    """Frozen ImageNet ResNet encoder for generic image embeddings."""

    def __init__(
        self,
        name: str = "resnet50",
        pretrained: bool = True,
        device: Optional[str] = None,
        image_size: int = 224,
    ) -> None:
        name = name.lower()
        if name not in _RESNET_SPECS:
            raise ValueError(f"Unsupported ResNet encoder: {name}")

        self.name = name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        builder, weights, dim = _RESNET_SPECS[name]
        self.output_dim = dim

        model = builder(weights=weights if pretrained else None)
        model.fc = torch.nn.Identity()
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
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(tensor).detach().cpu().numpy()[0]
        return l2_normalize(embedding)
