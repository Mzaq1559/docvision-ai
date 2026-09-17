"""
Document boundary detection.

Strategy:
  1. Edge-detect the (resized) image.
  2. Find external contours, sorted by area (largest first).
  3. For each candidate contour, approximate it as a polygon. If the
     approximation has exactly four points and covers a large-enough
     fraction of the image, treat it as the document.
  4. Handle rotated/skewed documents naturally: approxPolyDP works on the
     contour's shape, not its axis alignment, so a document photographed
     at an angle still reduces to a 4-point quadrilateral.
  5. If no 4-point contour is found (e.g. low contrast, cluttered
     background, torn/irregular edges), fall back gracefully:
       a. Use the largest contour's minimum-area bounding rectangle, if its
          area is reasonable.
       b. Otherwise, use the full image as the "document" so downstream
          steps (perspective transform becomes a no-op crop) never crash.

The corners are always returned in the OpenCV convention adopted here as
[top-left, top-right, bottom-right, bottom-left] via ``order_points``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from app.cv.preprocessing import detect_edges


@dataclass
class DetectionResult:
    corners: np.ndarray            # shape (4, 2), ordered TL, TR, BR, BL
    contour: Optional[np.ndarray]  # raw contour used (None if full-image fallback)
    used_fallback: bool
    confidence: str  # "high", "low", "fallback"


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype("float32")
    ordered = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]  # top-left has smallest sum
    ordered[2] = pts[np.argmax(s)]  # bottom-right has largest sum

    diff = np.diff(pts, axis=1)
    ordered[1] = pts[np.argmin(diff)]  # top-right has smallest difference
    ordered[3] = pts[np.argmax(diff)]  # bottom-left has largest difference

    return ordered


def _full_image_corners(image_shape: tuple) -> np.ndarray:
    height, width = image_shape[:2]
    return np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype="float32",
    )


def detect_document(
    image: np.ndarray,
    sensitivity: int = 50,
    min_area_ratio: float = 0.15,
    approx_epsilon_ratio: float = 0.02,
) -> DetectionResult:
    """Detect the document boundary in ``image`` (a BGR ndarray).

    Never raises on a "no document found" case; it always returns a usable
    (possibly fallback) set of corners so the pipeline can continue.
    """
    if image is None or image.size == 0:
        raise ValueError("detect_document received an empty image")

    height, width = image.shape[:2]
    image_area = float(height * width)

    edges = detect_edges(image, sensitivity=sensitivity)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:8]

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < image_area * min_area_ratio:
                continue
            perimeter = cv2.arcLength(contour, True)
            epsilon = approx_epsilon_ratio * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 4 and cv2.isContourConvex(approx):
                corners = order_points(approx)
                return DetectionResult(
                    corners=corners,
                    contour=contour,
                    used_fallback=False,
                    confidence="high",
                )

        # Fallback A: largest contour didn't reduce to a clean quadrilateral
        # (torn edge, cluttered background, partial occlusion). Use its
        # minimum-area rotated bounding rectangle instead -- still handles
        # rotation, just less precise than a true 4-point contour match.
        largest = contours[0]
        if cv2.contourArea(largest) >= image_area * (min_area_ratio * 0.5):
            rect = cv2.minAreaRect(largest)
            box = cv2.boxPoints(rect)
            corners = order_points(box)
            return DetectionResult(
                corners=corners,
                contour=largest,
                used_fallback=True,
                confidence="low",
            )

    # Fallback B: nothing usable was found at all. Treat the whole image as
    # the document rather than failing -- the perspective step below becomes
    # an identity crop, and the user still gets OCR + enhancement results.
    return DetectionResult(
        corners=_full_image_corners(image.shape),
        contour=None,
        used_fallback=True,
        confidence="fallback",
    )


def draw_detection(image: np.ndarray, result: DetectionResult) -> np.ndarray:
    """Return a copy of ``image`` with the detected boundary drawn on it."""
    overlay = image.copy()
    pts = result.corners.astype(int)
    color = (0, 200, 0) if not result.used_fallback else (0, 140, 255)
    cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=3)
    for point in pts:
        cv2.circle(overlay, tuple(point), 7, color, -1)
    return overlay
