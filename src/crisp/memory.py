from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class MemoryItem:
    embedding: np.ndarray
    label: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryBank:
    """Add-only memory bank for embeddings, labels, and metadata."""

    def __init__(self) -> None:
        self.items: List[MemoryItem] = []

    def add(
        self,
        embedding: np.ndarray,
        label: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        self.items.append(
            MemoryItem(
                embedding=emb,
                label=str(label),
                metadata=metadata or {},
            )
        )

    def extend(self, other: "MemoryBank") -> None:
        self.items.extend(other.items)

    def clear(self) -> None:
        self.items.clear()

    def as_matrix(self) -> np.ndarray:
        if not self.items:
            return np.empty((0, 0), dtype=np.float32)
        return np.stack([item.embedding for item in self.items]).astype(np.float32)

    def labels(self) -> List[str]:
        return [item.label for item in self.items]

    def metadata(self) -> List[Dict[str, Any]]:
        return [item.metadata for item in self.items]

    def get(self, index: int) -> MemoryItem:
        return self.items[index]

    def save(self, path: str) -> None:
        payload = {
            "items": self.items,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict) and "items" in payload:
            self.items = payload["items"]
        elif isinstance(payload, list):
            self.items = payload
        else:
            raise ValueError("Unsupported memory file format.")

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)
