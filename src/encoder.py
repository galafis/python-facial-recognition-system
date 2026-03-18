"""Face Encoding Module.

Generates 128-dimensional face embeddings using deep learning models.
Supports encoding caching for performance optimization.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

from src.detector import DetectedFace

logger = logging.getLogger(__name__)


class FaceEncoder:
    """Face encoding engine using deep learning embeddings.

    Generates 128-dimensional vectors representing facial features.

    Args:
        model: Encoding model - 'small' (5 landmarks) or 'large' (68 landmarks).
        num_jitters: Times to re-sample face for encoding (higher = more accurate, slower).
        cache_path: Path to cache encodings for known faces.
    """

    def __init__(
        self,
        model: str = "large",
        num_jitters: int = 1,
        cache_path: Optional[str] = None,
    ) -> None:
        self.model = model
        self.num_jitters = num_jitters
        self.cache_path = Path(cache_path) if cache_path else None
        self._encoder = None
        self._cache: Dict[str, NDArray] = {}
        self._initialize_encoder()

        if self.cache_path and self.cache_path.exists():
            self._load_cache()

        logger.info(f"FaceEncoder initialized with model='{model}'")

    def _initialize_encoder(self) -> None:
        """Initialize the encoding backend."""
        try:
            import face_recognition
            self._encoder = face_recognition
            logger.info("Using face_recognition library for encoding")
        except ImportError:
            logger.warning("face_recognition not available, using OpenCV DNN")
            self._encoder = None

    def encode(
        self, image: np.ndarray, faces: List[DetectedFace]
    ) -> List[NDArray]:
        """Generate encodings for detected faces.

        Args:
            image: BGR image as numpy array.
            faces: List of detected faces.

        Returns:
            List of 128-dimensional encoding vectors.
        """
        if not faces:
            return []

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        locations = [face.bbox for face in faces]

        if self._encoder is not None:
            encodings = self._encoder.face_encodings(
                rgb_image,
                known_face_locations=locations,
                num_jitters=self.num_jitters,
                model=self.model,
            )
        else:
            encodings = self._encode_opencv(rgb_image, faces)

        logger.info(f"Generated {len(encodings)} encoding(s)")
        return [np.array(enc) for enc in encodings]

    def _encode_opencv(self, image: np.ndarray, faces: List[DetectedFace]) -> List[NDArray]:
        """Fallback encoding using OpenCV DNN."""
        encodings = []
        for face in faces:
            top, right, bottom, left = face.bbox
            face_roi = image[max(0, top):bottom, max(0, left):right]
            if face_roi.size == 0:
                continue
            face_resized = cv2.resize(face_roi, (96, 96))
            blob = cv2.dnn.blobFromImage(
                face_resized, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=False
            )
            encoding = blob.flatten()[:128]
            norm = np.linalg.norm(encoding)
            if norm > 0:
                encoding = encoding / norm
            encodings.append(encoding)
        return encodings

    def encode_known_faces(
        self, faces_dir: str
    ) -> Dict[str, List[NDArray]]:
        """Encode all known faces from a directory structure.

        Expected structure:
            faces_dir/
                person_name_1/
                    image1.jpg
                    image2.jpg
                person_name_2/
                    image1.jpg

        Returns:
            Dictionary mapping person names to lists of encodings.
        """
        faces_path = Path(faces_dir)
        if not faces_path.exists():
            raise FileNotFoundError(f"Directory not found: {faces_dir}")

        known_encodings: Dict[str, List[NDArray]] = {}

        for person_dir in sorted(faces_path.iterdir()):
            if not person_dir.is_dir():
                continue

            person_name = person_dir.name
            encodings = []

            for img_path in sorted(person_dir.iterdir()):
                if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                    continue

                image = cv2.imread(str(img_path))
                if image is None:
                    logger.warning(f"Failed to load: {img_path}")
                    continue

                from src.detector import FaceDetector
                detector = FaceDetector()
                detected = detector.detect(image)

                if detected:
                    face_encodings = self.encode(image, detected[:1])
                    encodings.extend(face_encodings)
                    logger.debug(f"Encoded face from {img_path.name} for {person_name}")

            if encodings:
                known_encodings[person_name] = encodings
                logger.info(f"Encoded {len(encodings)} face(s) for '{person_name}'")

        self._cache = {k: v[0] for k, v in known_encodings.items() if v}
        if self.cache_path:
            self._save_cache()

        return known_encodings

    def _save_cache(self) -> None:
        """Save encodings cache to disk."""
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump(self._cache, f)
            logger.info(f"Cache saved to {self.cache_path}")

    def _load_cache(self) -> None:
        """Load encodings cache from disk."""
        if self.cache_path and self.cache_path.exists():
            with open(self.cache_path, "rb") as f:
                self._cache = pickle.load(f)
            logger.info(f"Cache loaded: {len(self._cache)} entries")
