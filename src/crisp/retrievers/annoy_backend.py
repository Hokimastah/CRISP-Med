from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..memory import MemoryBank
from ..encoders.base import l2_normalize
from .base import BaseRetriever


class AnnoyRetriever(BaseRetriever):
    """Approximate nearest neighbor retrieval using Annoy."""

    def __init__(self, n_trees: int = 10, metric: str = "angular") -> None:
        try:
            from annoy import AnnoyIndex
        except ImportError as exc:
            raise ImportError("Annoy backend requires annoy. Install with: pip install -e '.[annoy]'") from exc
        self.AnnoyIndex = AnnoyIndex
        self.n_trees = n_trees
        self.metric = metric
        self.index = None
        self.dim = None

    def build(self, memory: MemoryBank) -> None:
        if len(memory) == 0:
            self.index = None
            self.dim = None
            return
        matrix = memory.as_matrix()
        self.dim = matrix.shape[1]
        index = self.AnnoyIndex(self.dim, self.metric)
        for i, vector in enumerate(matrix):
            index.add_item(i, vector.tolist())
        index.build(self.n_trees)
        self.index = index

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

        query = l2_normalize(query_embedding).tolist()
        k = min(int(top_k), len(memory))
        indices, distances = self.index.get_nns_by_vector(query, k, include_distances=True)

        neighbors = []
        for idx, distance in zip(indices, distances):
            item = memory.get(int(idx))
            if self.metric == "angular":
                similarity = 1.0 - (float(distance) ** 2) / 2.0
            else:
                similarity = -float(distance)
            neighbors.append({
                "index": int(idx),
                "label": item.label,
                "similarity": float(similarity),
                "metadata": item.metadata,
            })
        return neighbors
