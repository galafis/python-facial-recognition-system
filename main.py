"""Facial Recognition System - Main Entry Point.

Command-line interface for face detection, encoding and matching.
Supports registration of known faces and identification of unknown faces.

Usage:
    python main.py --mode register --input data/known_faces
    python main.py --mode identify --input data/input/photo.jpg
    python main.py --mode verify --input face1.jpg --compare face2.jpg
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2

from src.detector import FaceDetector
from src.encoder import FaceEncoder
from src.matcher import FaceMatcher
from src.utils import ensure_directories, load_config, setup_logging

logger = logging.getLogger(__name__)


def create_pipeline(config: dict) -> tuple:
    """Create the face recognition pipeline from config."""
    detector = FaceDetector(
        model=config["detection"]["model"],
        upsample_times=config["detection"]["upsample_times"],
        min_face_size=config["detection"]["min_face_size"],
        confidence_threshold=config["detection"]["confidence_threshold"],
    )
    encoder = FaceEncoder(
        model=config["encoding"]["model"],
        num_jitters=config["encoding"]["num_jitters"],
        cache_path=config["paths"].get("encodings_cache"),
    )
    matcher = FaceMatcher(
        tolerance=config["matching"]["tolerance"],
        algorithm=config["matching"]["algorithm"],
        top_k=config["matching"]["top_k"],
    )
    return detector, encoder, matcher


def register_faces(config: dict) -> None:
    """Register known faces from directory."""
    detector, encoder, matcher = create_pipeline(config)
    faces_dir = config["paths"]["known_faces_dir"]

    logger.info(f"Registering known faces from: {faces_dir}")
    known_encodings = encoder.encode_known_faces(faces_dir)

    if known_encodings:
        logger.info(f"Successfully registered {len(known_encodings)} people")
        for name, encs in known_encodings.items():
            logger.info(f"  - {name}: {len(encs)} encoding(s)")
    else:
        logger.warning("No faces found to register")


def identify_faces(config: dict, input_path: str) -> None:
    """Identify faces in an image."""
    detector, encoder, matcher = create_pipeline(config)

    # Load known encodings
    faces_dir = config["paths"]["known_faces_dir"]
    known_encodings = encoder.encode_known_faces(faces_dir)
    matcher.register_faces(known_encodings)

    # Process input image
    image = cv2.imread(input_path)
    if image is None:
        logger.error(f"Failed to load image: {input_path}")
        sys.exit(1)

    faces = detector.detect(image)
    logger.info(f"Detected {len(faces)} face(s) in {input_path}")

    if not faces:
        logger.info("No faces detected")
        return

    encodings = encoder.encode(image, faces)
    results = matcher.batch_match(encodings)

    # Annotate and save output
    output = image.copy()
    for face, result in zip(faces, results):
        top, right, bottom, left = face.bbox
        color = (0, 255, 0) if result.matched else (0, 0, 255)
        cv2.rectangle(output, (left, top), (right, bottom), color, 2)
        label = f"{result.name} ({result.confidence:.2f})"
        cv2.putText(
            output, label, (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )
        logger.info(f"Face: {result}")

    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"result_{Path(input_path).name}"
    cv2.imwrite(str(output_path), output)
    logger.info(f"Result saved to: {output_path}")


def verify_faces(config: dict, path1: str, path2: str) -> None:
    """Verify if two images contain the same person."""
    detector, encoder, matcher = create_pipeline(config)

    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    if img1 is None or img2 is None:
        logger.error("Failed to load one or both images")
        sys.exit(1)

    faces1 = detector.detect(img1)
    faces2 = detector.detect(img2)

    if not faces1 or not faces2:
        logger.error("Could not detect faces in one or both images")
        sys.exit(1)

    enc1 = encoder.encode(img1, faces1[:1])[0]
    enc2 = encoder.encode(img2, faces2[:1])[0]

    is_match, distance = matcher.verify(enc1, enc2)

    status = "SAME PERSON" if is_match else "DIFFERENT PEOPLE"
    logger.info(f"Verification result: {status} (distance: {distance:.4f})")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Facial Recognition System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["register", "identify", "verify"],
        default="identify",
        help="Operation mode (default: identify)",
    )
    parser.add_argument(
        "--input", "-i",
        help="Input image path or directory",
    )
    parser.add_argument(
        "--compare", "-c",
        help="Second image for verification mode",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    config = load_config(args.config)

    log_config = config.get("logging", {})
    setup_logging(
        level=log_config.get("level", "INFO"),
        log_file=log_config.get("file"),
    )

    ensure_directories(config)
    logger.info(f"Starting Facial Recognition System in '{args.mode}' mode")

    if args.mode == "register":
        register_faces(config)
    elif args.mode == "identify":
        if not args.input:
            logger.error("--input required for identify mode")
            sys.exit(1)
        identify_faces(config, args.input)
    elif args.mode == "verify":
        if not args.input or not args.compare:
            logger.error("--input and --compare required for verify mode")
            sys.exit(1)
        verify_faces(config, args.input, args.compare)

    logger.info("Done.")


if __name__ == "__main__":
    main()
