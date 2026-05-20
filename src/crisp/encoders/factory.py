from __future__ import annotations

from typing import Any, Dict, Optional

from .base import BaseEncoder

RESNET_NAMES = {"resnet18", "resnet34", "resnet50", "resnet101", "resnet152"}
MEDICAL_NAMES = {
    "medical_resnet18",
    "medical_resnet34",
    "medical_resnet50",
    "medical_densenet121",
    "medical_efficientnet_b0",
}


def build_encoder(
    encoder: str = "resnet50",
    device: Optional[str] = None,
    pretrained: bool = True,
    encoder_kwargs: Optional[Dict[str, Any]] = None,
) -> BaseEncoder:
    name = encoder.lower()
    kwargs = encoder_kwargs or {}

    if name in RESNET_NAMES:
        from .resnet import ResNetEncoder

        return ResNetEncoder(name=name, pretrained=pretrained, device=device, **kwargs)

    if name == "clip":
        from .clip_encoder import CLIPEncoder

        return CLIPEncoder(device=device, pretrained=kwargs.pop("pretrained", "openai"), **kwargs)

    if name == "arcface":
        from .arcface import ArcFaceEncoder

        return ArcFaceEncoder(device=device, **kwargs)

    if name in MEDICAL_NAMES:
        from .medical import MedicalImageEncoder

        return MedicalImageEncoder(name=name, pretrained=pretrained, device=device, **kwargs)

    supported = sorted(list(RESNET_NAMES | MEDICAL_NAMES | {"clip", "arcface"}))
    raise ValueError(f"Unsupported encoder '{encoder}'. Supported encoders: {supported}")
