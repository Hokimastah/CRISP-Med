from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..memory import MemoryBank
from ..encoders.base import l2_normalize
from .base import BaseRetriever


class NumpyRetriever(BaseRetriever):
    """Exact brute-force cosine retrieval using NumPy."""

    def __init__(self) -> None:
        self.matrix: np.ndarray | None = None

    def build(self, memory: MemoryBank) -> None:
        self.matrix = memory.as_matrix()

    def search(
        self,
        query_embedding: np.ndarray,
        memory: MemoryBank,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        if len(memory) == 0:
            return []
        if self.matrix is None or len(self.matrix) != len(memory):
            self.build(memory)

        query = l2_normalize(query_embedding)
        similarities = self.matrix @ query
        k = min(int(top_k), len(memory))
        order = np.argsort(-similarities)[:k]

        neighbors = []
        for idx in order:
            item = memory.get(int(idx))
            neighbors.append({
                "index": int(idx),
                "label": item.label,
                "similarity": float(similarities[idx]),
                "metadata": item.metadata,
            })
        return neighbors
