import numpy as np

from crisp.memory import MemoryBank
from crisp.retrievers import build_retriever


def test_numpy_retriever():
    memory = MemoryBank()
    memory.add(np.array([1.0, 0.0]), "a")
    memory.add(np.array([0.0, 1.0]), "b")

    retriever = build_retriever("numpy")
    retriever.build(memory)
    result = retriever.search(np.array([1.0, 0.0]), memory, top_k=1)
    assert result[0]["label"] == "a"
