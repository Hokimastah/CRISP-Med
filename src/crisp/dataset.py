from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from torch.utils.data import Dataset

from .encoders.medical import MedicalPreprocessor
from .utils import list_images


class FolderMedicalDataset(Dataset):
    """
    Folder-based medical image classification dataset.

    Format:
        dataset/
        ├── class_a/
        │   ├── image_1.png
        │   └── image_2.dcm
        └── class_b/
            ├── image_3.png
            └── image_4.dcm
    """

    def __init__(
        self,
        root: str,
        image_size: int = 224,
        intensity_mode: str = "percentile",
        normalize: str = "imagenet",
        class_to_idx: Optional[Dict[str, int]] = None,
    ) -> None:
        self.root = Path(root)

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset folder not found: {root}")

        if class_to_idx is None:
            classes = sorted([p.name for p in self.root.iterdir() if p.is_dir()])

            if not classes:
                raise ValueError(f"No class folders found in {root}")

            self.class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
        else:
            self.class_to_idx = dict(class_to_idx)

        self.idx_to_class = {idx: class_name for class_name, idx in self.class_to_idx.items()}

        samples: List[Tuple[str, int]] = []

        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.root / class_name

            if not class_dir.exists():
                continue

            for image_path in list_images(str(class_dir)):
                samples.append((str(image_path), class_idx))

        if not samples:
            raise ValueError(f"No supported image files found in {root}")

        self.samples = samples

        self.preprocessor = MedicalPreprocessor(
            image_size=image_size,
            intensity_mode=intensity_mode,
            normalize=normalize,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        tensor = self.preprocessor(image_path)
        return tensor, label

    @property
    def classes(self) -> List[str]:
        return [self.idx_to_class[i] for i in sorted(self.idx_to_class)]