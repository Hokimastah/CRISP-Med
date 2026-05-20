from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..memory import MemoryBank
from ..encoders.base import l2_normalize
from .base import BaseRetriever


class FaissRetriever(BaseRetriever):
    """Dense vector similarity search using FAISS IndexFlatIP."""

    def __init__(self) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise ImportError("FAISS backend requires faiss-cpu. Install with: pip install -e '.[faiss]'") from exc
        self.faiss = faiss
        self.index = None
        self.dim = None

    def build(self, memory: MemoryBank) -> None:
        if len(memory) == 0:
            self.index = None
            self.dim = None
            return
        matrix = memory.as_matrix().astype(np.float32)
        self.dim = matrix.shape[1]
        self.index = self.faiss.IndexFlatIP(self.dim)
        self.index.add(matrix)

    def search(
        self,
        query_embedding: np.ndarray,
        memory: MemoryBank,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        if len(memory) == 0:
            return []
        if self.index is None:
            self.build(memory)
        query = l2_normalize(query_embedding).astype(np.float32).reshape(1, -1)
        k = min(int(top_k), len(memory))
        sims, indices = self.index.search(query, k)

        neighbors = []
        for idx, sim in zip(indices[0], sims[0]):
            if idx < 0:
                continue
            item = memory.get(int(idx))
            neighbors.append({
                "index": int(idx),
                "label": item.label,
                "similarity": float(sim),
                "metadata": item.metadata,
            })
        return neighbors
