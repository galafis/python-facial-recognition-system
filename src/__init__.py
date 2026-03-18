"""Facial Recognition System - Python package for face detection, encoding and matching."""

__version__ = "1.0.0"
__author__ = "galafis"

from src.detector import FaceDetector
from src.encoder import FaceEncoder
from src.matcher import FaceMatcher

__all__ = ["FaceDetector", "FaceEncoder", "FaceMatcher"]
