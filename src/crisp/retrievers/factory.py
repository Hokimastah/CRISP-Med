from __future__ import annotations

from typing import Any, Dict, Optional

from .base import BaseRetriever
from .numpy_backend import NumpyRetriever


def build_retriever(
    retriever: str = "numpy",
    retriever_kwargs: Optional[Dict[str, Any]] = None,
) -> BaseRetriever:
    name = retriever.lower()
    kwargs = retriever_kwargs or {}

    if name == "numpy":
        return NumpyRetriever(**kwargs)
    if name == "annoy":
        from .annoy_backend import AnnoyRetriever

        return AnnoyRetriever(**kwargs)
    if name == "faiss":
        from .faiss_backend import FaissRetriever

        return FaissRetriever(**kwargs)

    raise ValueError("retriever must be one of: numpy, annoy, faiss")
