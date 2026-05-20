from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import numpy as np

from .memory import MemoryBank, MemoryItem


def herding_memory(memory: MemoryBank, max_per_class: int) -> MemoryBank:
    """
    Select representative embeddings per class using a simple herding strategy.

    For each class, the function computes the class centroid and keeps samples
    closest to that centroid. This reduces memory size while preserving the
    most representative examples.
    """
    if max_per_class <= 0:
        raise ValueError("max_per_class must be greater than zero.")

    grouped: Dict[str, List[MemoryItem]] = defaultdict(list)
    for item in memory:
        grouped[item.label].append(item)

    new_memory = MemoryBank()
    for label, items in grouped.items():
        if len(items) <= max_per_class:
            for item in items:
                new_memory.add(item.embedding, item.label, item.metadata)
            continue

        matrix = np.stack([item.embedding for item in items]).astype(np.float32)
        centroid = matrix.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        similarities = matrix @ centroid
        selected_indices = np.argsort(-similarities)[:max_per_class]

        for idx in selected_indices:
            item = items[int(idx)]
            new_memory.add(item.embedding, item.label, item.metadata)

    return new_memory
