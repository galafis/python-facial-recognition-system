"""Face Detection Module.

Provides face detection using OpenCV cascades and dlib HOG/CNN models.
Supports multiple detection backends with configurable parameters.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DetectedFace:
    """Represents a detected face with bounding box and confidence."""
    bbox: Tuple[int, int, int, int]  # (top, right, bottom, left)
    confidence: float
    landmarks: Optional[dict] = None

    @property
    def width(self) -> int:
        return self.bbox[1] - self.bbox[3]

    @property
    def height(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Tuple[int, int]:
        return (
            (self.bbox[3] + self.bbox[1]) // 2,
            (self.bbox[0] + self.bbox[2]) // 2,
        )


class FaceDetector:
    """Multi-backend face detection engine.

    Supports HOG (fast, CPU) and CNN (accurate, GPU) detection models.

    Args:
        model: Detection model - 'hog' or 'cnn'.
        upsample_times: Number of times to upsample image.
        min_face_size: Minimum face size in pixels.
        confidence_threshold: Minimum detection confidence.
    """

    SUPPORTED_MODELS = ("hog", "cnn")

    def __init__(
        self,
        model: str = "hog",
        upsample_times: int = 1,
        min_face_size: int = 20,
        confidence_threshold: float = 0.6,
    ) -> None:
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model must be one of {self.SUPPORTED_MODELS}")

        self.model = model
        self.upsample_times = upsample_times
        self.min_face_size = min_face_size
        self.confidence_threshold = confidence_threshold
        self._detector = None
        self._initialize_detector()
        logger.info(f"FaceDetector initialized with model='{model}'")

    def _initialize_detector(self) -> None:
        """Initialize the detection backend."""
        try:
            import dlib
            if self.model == "cnn":
                model_path = Path("models/mmod_human_face_detector.dat")
                if model_path.exists():
                    self._detector = dlib.cnn_face_detection_model_v1(str(model_path))
                else:
                    logger.warning("CNN model not found, falling back to HOG")
                    self._detector = dlib.get_frontal_face_detector()
                    self.model = "hog"
            else:
                self._detector = dlib.get_frontal_face_detector()
        except ImportError:
            logger.warning("dlib not available, using OpenCV Haar cascade")
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._detector = cv2.CascadeClassifier(cascade_path)

    def detect(self, image: np.ndarray) -> List[DetectedFace]:
        """Detect faces in an image.

        Args:
            image: BGR image as numpy array.

        Returns:
            List of DetectedFace objects.
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid image provided")

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if isinstance(self._detector, cv2.CascadeClassifier):
            return self._detect_opencv(image)
        else:
            return self._detect_dlib(rgb_image)

    def _detect_dlib(self, rgb_image: np.ndarray) -> List[DetectedFace]:
        """Detect faces using dlib."""
        import dlib

        detections = self._detector(rgb_image, self.upsample_times)
        faces = []

        for detection in detections:
            if hasattr(detection, "rect"):
                rect = detection.rect
                confidence = detection.confidence
            else:
                rect = detection
                confidence = 1.0

            if confidence < self.confidence_threshold:
                continue

            bbox = (rect.top(), rect.right(), rect.bottom(), rect.left())
            face = DetectedFace(bbox=bbox, confidence=confidence)

            if face.width >= self.min_face_size and face.height >= self.min_face_size:
                faces.append(face)

        logger.info(f"Detected {len(faces)} face(s) using dlib {self.model}")
        return faces

    def _detect_opencv(self, image: np.ndarray) -> List[DetectedFace]:
        """Detect faces using OpenCV Haar cascade."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        rects = self._detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size),
        )

        faces = []
        for (x, y, w, h) in rects:
            bbox = (y, x + w, y + h, x)  # Convert to (top, right, bottom, left)
            faces.append(DetectedFace(bbox=bbox, confidence=1.0))

        logger.info(f"Detected {len(faces)} face(s) using OpenCV Haar cascade")
        return faces

    def detect_from_file(self, image_path: str) -> List[DetectedFace]:
        """Detect faces from an image file."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        return self.detect(image)

    def draw_detections(
        self,
        image: np.ndarray,
        faces: List[DetectedFace],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw bounding boxes on detected faces."""
        output = image.copy()
        for face in faces:
            top, right, bottom, left = face.bbox
            cv2.rectangle(output, (left, top), (right, bottom), color, thickness)
            label = f"{face.confidence:.2f}"
            cv2.putText(
                output, label, (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
            )
        return output
