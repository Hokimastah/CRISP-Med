from crisp.voting import majority_vote, weighted_vote


def test_weighted_vote():
    neighbors = [
        {"label": "a", "similarity": 0.8},
        {"label": "b", "similarity": 0.7},
        {"label": "a", "similarity": 0.4},
    ]
    result = weighted_vote(neighbors)
    assert result["predicted_label"] == "a"


def test_majority_vote():
    neighbors = [
        {"label": "a", "similarity": 0.8},
        {"label": "b", "similarity": 0.7},
        {"label": "b", "similarity": 0.4},
    ]
    result = majority_vote(neighbors)
    assert result["predicted_label"] == "b"
