"""Tests for the Facial Recognition System."""

import numpy as np
import pytest

from src.detector import DetectedFace, FaceDetector
from src.encoder import FaceEncoder
from src.matcher import FaceMatcher, MatchResult
from src.utils import get_default_config, load_config, preprocess_image


class TestDetectedFace:
    """Tests for DetectedFace dataclass."""

    def test_create_detected_face(self):
        face = DetectedFace(bbox=(10, 110, 110, 10), confidence=0.95)
        assert face.bbox == (10, 110, 110, 10)
        assert face.confidence == 0.95

    def test_width(self):
        face = DetectedFace(bbox=(10, 110, 110, 10), confidence=0.9)
        assert face.width == 100

    def test_height(self):
        face = DetectedFace(bbox=(10, 110, 110, 10), confidence=0.9)
        assert face.height == 100

    def test_area(self):
        face = DetectedFace(bbox=(10, 110, 110, 10), confidence=0.9)
        assert face.area == 10000

    def test_center(self):
        face = DetectedFace(bbox=(10, 110, 110, 10), confidence=0.9)
        assert face.center == (60, 60)


class TestFaceDetector:
    """Tests for FaceDetector."""

    def test_init_default(self):
        detector = FaceDetector()
        assert detector.model in ("hog", "cnn")
        assert detector.min_face_size == 20

    def test_invalid_model(self):
        with pytest.raises(ValueError):
            FaceDetector(model="invalid")

    def test_detect_empty_image(self):
        detector = FaceDetector()
        with pytest.raises(ValueError):
            detector.detect(np.array([]))

    def test_detect_synthetic_image(self):
        detector = FaceDetector()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = detector.detect(image)
        assert isinstance(faces, list)

    def test_detect_from_nonexistent_file(self):
        detector = FaceDetector()
        with pytest.raises(FileNotFoundError):
            detector.detect_from_file("nonexistent.jpg")


class TestFaceMatcher:
    """Tests for FaceMatcher."""

    def test_init_default(self):
        matcher = FaceMatcher()
        assert matcher.tolerance == 0.6
        assert matcher.algorithm == "euclidean"

    def test_invalid_algorithm(self):
        with pytest.raises(ValueError):
            FaceMatcher(algorithm="invalid")

    def test_match_no_known_faces(self):
        matcher = FaceMatcher()
        encoding = np.random.rand(128)
        results = matcher.match(encoding)
        assert len(results) == 1
        assert results[0].name == "Unknown"

    def test_identify_with_known_faces(self):
        matcher = FaceMatcher(tolerance=0.6)
        known = {
            "Alice": [np.random.rand(128)],
            "Bob": [np.random.rand(128)],
        }
        matcher.register_faces(known)
        assert matcher.registered_count == 2

    def test_verify_same_encoding(self):
        matcher = FaceMatcher(tolerance=0.6)
        encoding = np.random.rand(128)
        is_match, distance = matcher.verify(encoding, encoding)
        assert is_match is True
        assert distance == 0.0

    def test_verify_different_encodings(self):
        matcher = FaceMatcher(tolerance=0.1)
        enc1 = np.ones(128)
        enc2 = np.zeros(128)
        is_match, _ = matcher.verify(enc1, enc2)
        assert is_match is False

    def test_batch_match(self):
        matcher = FaceMatcher()
        known = {"Alice": [np.random.rand(128)]}
        matcher.register_faces(known)
        encodings = [np.random.rand(128) for _ in range(3)]
        results = matcher.batch_match(encodings)
        assert len(results) == 3

    def test_match_result_repr(self):
        result = MatchResult("Alice", 0.3, 0.75, True)
        assert "MATCH" in repr(result)


class TestUtils:
    """Tests for utility functions."""

    def test_default_config(self):
        config = get_default_config()
        assert "detection" in config
        assert "encoding" in config
        assert "matching" in config
        assert "paths" in config

    def test_load_config_missing(self):
        config = load_config("nonexistent.yaml")
        assert isinstance(config, dict)
        assert "detection" in config

    def test_preprocess_image(self):
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        processed = preprocess_image(image, target_size=(50, 50))
        assert processed.shape == (50, 50, 3)

    def test_preprocess_image_normalize(self):
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        processed = preprocess_image(image, normalize=True)
        assert processed.max() <= 1.0

    def test_preprocess_invalid_image(self):
        with pytest.raises(ValueError):
            preprocess_image(np.array([]))
