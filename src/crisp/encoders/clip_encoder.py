from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from PIL import Image

from .base import BaseEncoder, l2_normalize


class CLIPEncoder(BaseEncoder):
    """Frozen CLIP image encoder using open_clip_torch."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str | bool = "openai",
        device: Optional[str] = None,
    ) -> None:
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "CLIP encoder requires open_clip_torch. Install with: pip install -e '.[clip]'"
            ) from exc

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        pretrained_tag = pretrained if isinstance(pretrained, str) else "openai"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained_tag,
        )
        self.model.eval().to(self.device)
        self.model_name = model_name

    def encode_path(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_image(tensor).detach().cpu().numpy()[0]
        return l2_normalize(embedding)
