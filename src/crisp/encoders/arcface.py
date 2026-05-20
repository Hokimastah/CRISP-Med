from __future__ import annotations

from typing import Optional

import numpy as np

from .base import BaseEncoder, l2_normalize


class ArcFaceEncoder(BaseEncoder):
    """
    Face-specific encoder using InsightFace ArcFace embeddings.

    Use this encoder for face recognition datasets. It performs face detection
    and alignment internally through InsightFace FaceAnalysis.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        model_name: str = "buffalo_l",
        det_size: tuple[int, int] = (640, 640),
        face_selection: str = "largest",
    ) -> None:
        try:
            import cv2
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise ImportError(
                "ArcFace encoder requires insightface, opencv-python, and onnxruntime. "
                "Install with: pip install -e '.[arcface]'"
            ) from exc

        self.cv2 = cv2
        self.device = device or "cpu"
        self.model_name = model_name
        self.det_size = det_size
        self.face_selection = face_selection

        providers = ["CPUExecutionProvider"]
        ctx_id = -1
        if self.device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ctx_id = 0

        self.app = FaceAnalysis(name=model_name, providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
        self.output_dim = 512

    def _select_face(self, faces):
        if not faces:
            return None
        if self.face_selection == "first":
            return faces[0]
        if self.face_selection == "largest":
            return max(
                faces,
                key=lambda f: max(0.0, float(f.bbox[2] - f.bbox[0]))
                * max(0.0, float(f.bbox[3] - f.bbox[1])),
            )
        raise ValueError("face_selection must be either 'largest' or 'first'.")

    def encode_path(self, image_path: str) -> np.ndarray:
        image_bgr = self.cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        faces = self.app.get(image_bgr)
        face = self._select_face(faces)
        if face is None:
            raise ValueError(f"No face detected in image: {image_path}")

        embedding = np.asarray(face.embedding, dtype=np.float32)
        return l2_normalize(embedding)
