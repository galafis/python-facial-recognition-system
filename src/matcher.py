"""Face Matching Module.

Compares face encodings to identify and verify individuals.
Supports euclidean and cosine distance metrics.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cosine, euclidean

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Represents a face matching result."""
    name: str
    distance: float
    confidence: float
    matched: bool

    def __repr__(self) -> str:
        status = "MATCH" if self.matched else "NO MATCH"
        return f"MatchResult({self.name}, {status}, conf={self.confidence:.3f})"


class FaceMatcher:
    """Face matching engine using distance-based comparison.

    Args:
        tolerance: Maximum distance threshold for a match.
        algorithm: Distance algorithm - 'euclidean' or 'cosine'.
        top_k: Number of top matches to return.
    """

    SUPPORTED_ALGORITHMS = ("euclidean", "cosine")

    def __init__(
        self,
        tolerance: float = 0.6,
        algorithm: str = "euclidean",
        top_k: int = 5,
    ) -> None:
        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(f"Algorithm must be one of {self.SUPPORTED_ALGORITHMS}")

        self.tolerance = tolerance
        self.algorithm = algorithm
        self.top_k = top_k
        self._known_encodings: Dict[str, List[NDArray]] = {}
        logger.info(
            f"FaceMatcher initialized: algorithm='{algorithm}', tolerance={tolerance}"
        )

    def register_faces(
        self, known_encodings: Dict[str, List[NDArray]]
    ) -> None:
        """Register known face encodings for matching."""
        self._known_encodings = known_encodings
        total = sum(len(v) for v in known_encodings.values())
        logger.info(
            f"Registered {len(known_encodings)} people with {total} encodings"
        )

    def match(
        self, encoding: NDArray
    ) -> List[MatchResult]:
        """Match a face encoding against known faces.

        Args:
            encoding: 128-dimensional face encoding vector.

        Returns:
            List of MatchResult sorted by distance (best first).
        """
        if not self._known_encodings:
            logger.warning("No known faces registered")
            return [MatchResult("Unknown", float("inf"), 0.0, False)]

        results = []

        for name, encodings in self._known_encodings.items():
            distances = [
                self._compute_distance(encoding, known_enc)
                for known_enc in encodings
            ]
            min_distance = min(distances)
            confidence = self._distance_to_confidence(min_distance)
            matched = min_distance <= self.tolerance

            results.append(
                MatchResult(
                    name=name,
                    distance=min_distance,
                    confidence=confidence,
                    matched=matched,
                )
            )

        results.sort(key=lambda r: r.distance)
        return results[: self.top_k]

    def identify(
        self, encoding: NDArray
    ) -> MatchResult:
        """Identify a face - return best match or Unknown."""
        matches = self.match(encoding)
        if matches and matches[0].matched:
            return matches[0]
        return MatchResult("Unknown", float("inf"), 0.0, False)

    def verify(
        self, encoding1: NDArray, encoding2: NDArray
    ) -> Tuple[bool, float]:
        """Verify if two encodings belong to the same person.

        Returns:
            Tuple of (is_same_person, distance).
        """
        distance = self._compute_distance(encoding1, encoding2)
        is_match = distance <= self.tolerance
        logger.info(f"Verification: distance={distance:.4f}, match={is_match}")
        return is_match, distance

    def _compute_distance(self, enc1: NDArray, enc2: NDArray) -> float:
        """Compute distance between two encodings."""
        if self.algorithm == "cosine":
            return float(cosine(enc1, enc2))
        return float(euclidean(enc1, enc2))

    def _distance_to_confidence(self, distance: float) -> float:
        """Convert distance to confidence score (0-1)."""
        if self.algorithm == "cosine":
            return max(0.0, 1.0 - distance)
        return max(0.0, 1.0 - (distance / (2 * self.tolerance)))

    def batch_match(
        self, encodings: List[NDArray]
    ) -> List[MatchResult]:
        """Match multiple encodings at once."""
        return [self.identify(enc) for enc in encodings]

    @property
    def registered_count(self) -> int:
        """Number of registered people."""
        return len(self._known_encodings)
