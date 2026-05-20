from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseEncoder(ABC):
    """Base interface for all CRISP image encoders."""

    @abstractmethod
    def encode_path(self, image_path: str) -> np.ndarray:
        """Return a 1D L2-normalized embedding for an image path."""
        raise NotImplementedError



def l2_normalize(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vector)
    if norm < eps:
        return vector
    return (vector / norm).astype(np.float32)
