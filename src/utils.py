"""Utility functions for the Facial Recognition System.

Provides image preprocessing, logging setup, and configuration loading.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    fmt: str = "%%(asctime)s - %%(name)s - %%(levelname)s - %%(message)s",
) -> None:
    """Configure logging for the application."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(level=log_level, format=fmt, handlers=handlers)
    logger.info(f"Logging configured: level={level}")


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Config not found: {config_path}, using defaults")
        return get_default_config()

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Configuration loaded from {config_path}")
    return config


def get_default_config() -> Dict[str, Any]:
    """Return default configuration."""
    return {
        "detection": {
            "model": "hog",
            "upsample_times": 1,
            "min_face_size": 20,
            "confidence_threshold": 0.6,
        },
        "encoding": {
            "model": "large",
            "num_jitters": 1,
        },
        "matching": {
            "tolerance": 0.6,
            "algorithm": "euclidean",
            "top_k": 5,
        },
        "paths": {
            "known_faces_dir": "data/known_faces",
            "input_dir": "data/input",
            "output_dir": "data/output",
            "encodings_cache": "data/encodings.pkl",
        },
        "logging": {
            "level": "INFO",
        },
    }


def preprocess_image(
    image: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
    normalize: bool = False,
) -> np.ndarray:
    """Preprocess image for face detection/encoding."""
    if image is None or image.size == 0:
        raise ValueError("Invalid image")

    processed = image.copy()

    if target_size:
        processed = cv2.resize(processed, target_size)

    if normalize:
        processed = processed.astype(np.float32) / 255.0

    return processed


def ensure_directories(config: Dict[str, Any]) -> None:
    """Create necessary directories from configuration."""
    paths = config.get("paths", {})
    for key, path_str in paths.items():
        if key.endswith("_dir"):
            Path(path_str).mkdir(parents=True, exist_ok=True)
            logger.debug(f"Directory ensured: {path_str}")

    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)


def validate_image(image_path: str) -> bool:
    """Validate that a file is a readable image."""
    path = Path(image_path)
    if not path.exists():
        return False
    if path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
        return False
    img = cv2.imread(str(path))
    return img is not None and img.size > 0
