from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List


def weighted_vote(neighbors: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores = defaultdict(float)
    for neighbor in neighbors:
        label = neighbor["label"]
        similarity = float(neighbor.get("similarity", 0.0))
        scores[label] += similarity

    scores_dict = dict(scores)
    predicted_label = max(scores_dict, key=scores_dict.get) if scores_dict else None
    return {
        "predicted_label": predicted_label,
        "scores": scores_dict,
    }


def majority_vote(neighbors: List[Dict[str, Any]]) -> Dict[str, Any]:
    counter = Counter(neighbor["label"] for neighbor in neighbors)
    scores_dict = dict(counter)
    predicted_label = counter.most_common(1)[0][0] if counter else None
    return {
        "predicted_label": predicted_label,
        "scores": scores_dict,
    }
