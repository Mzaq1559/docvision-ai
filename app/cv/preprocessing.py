"""
Preprocessing steps that feed the document detector: grayscale conversion,
blurring, and edge detection with a sensitivity control mapped to Canny
thresholds.
"""

from __future__ import annotations

import cv2
import numpy as np


def to_blurred_gray(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)


def sensitivity_to_canny_thresholds(sensitivity: int) -> tuple:
    """Map a 0-100 user-facing "detection sensitivity" slider to a pair of
    Canny (low, high) thresholds. Higher sensitivity -> lower thresholds
    -> more edges detected (useful for faint document boundaries).
    """
    sensitivity = max(0, min(100, int(sensitivity)))
    # At sensitivity=0 -> thresholds (150, 250); at sensitivity=100 -> (20, 80)
    low = int(150 - sensitivity * 1.3)
    high = int(250 - sensitivity * 1.7)
    low = max(10, low)
    high = max(low + 20, high)
    return low, high


def detect_edges(image: np.ndarray, sensitivity: int = 50) -> np.ndarray:
    """Full edge-detection pipeline: blur -> Canny -> dilate (to close small
    gaps in the document boundary so contour detection is more robust).
    """
    blurred = to_blurred_gray(image)
    low, high = sensitivity_to_canny_thresholds(sensitivity)
    edges = cv2.Canny(blurred, low, high)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
    return edges
