import numpy as np

from crisp.memory import MemoryBank


def test_memory_add_len():
    memory = MemoryBank()
    memory.add(np.array([1.0, 0.0]), "a", {"path": "x.jpg"})
    assert len(memory) == 1
    assert memory.labels() == ["a"]
