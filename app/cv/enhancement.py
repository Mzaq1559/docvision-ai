"""
Document readability enhancement.

Pipeline (each step optional / strength-controlled):
    grayscale -> denoise -> shadow reduction -> contrast (CLAHE) -> thresholding

All operations are classic, CPU-cheap OpenCV/NumPy -- no deep-learning
models -- which keeps this comfortably fast on Streamlit Community Cloud's
shared CPU instances.
"""

from __future__ import annotations

import cv2
import numpy as np

VALID_THRESHOLD_METHODS = ("adaptive_gaussian", "adaptive_mean", "otsu", "none")


def _strength_to_denoise_h(strength: int) -> float:
    # h controls filter strength in fastNlMeansDenoising; higher = smoother
    # but slower and can blur small text, so keep the range conservative.
    strength = max(0, min(100, int(strength)))
    return 3.0 + (strength / 100.0) * 7.0  # range roughly [3, 10]


def _strength_to_clip_limit(strength: int) -> float:
    strength = max(0, min(100, int(strength)))
    return 1.0 + (strength / 100.0) * 3.0  # range roughly [1, 4]


def denoise(gray: np.ndarray, strength: int = 50) -> np.ndarray:
    h = _strength_to_denoise_h(strength)
    return cv2.fastNlMeansDenoising(gray, None, h=h, templateWindowSize=7, searchWindowSize=21)


def enhance_contrast(gray: np.ndarray, strength: int = 50) -> np.ndarray:
    clip_limit = _strength_to_clip_limit(strength)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(gray)


def reduce_shadows(gray: np.ndarray) -> np.ndarray:
    """Estimate and remove a smooth background/shadow gradient.

    Approach: estimate the local background with a large-kernel dilation +
    median blur, then divide the original by that background so uneven
    illumination is normalized out. This is a well-known, lightweight
    technique that works reasonably well on document photos.
    """
    dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
    background = cv2.medianBlur(dilated, 21)
    # Avoid division by zero; background is >= 1 after the operations above.
    diff = 255 - cv2.absdiff(gray, background)
    normalized = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)


def apply_threshold(gray: np.ndarray, method: str = "adaptive_gaussian") -> np.ndarray:
    if method not in VALID_THRESHOLD_METHODS:
        method = "adaptive_gaussian"

    if method == "none":
        return gray
    if method == "otsu":
        _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return result
    if method == "adaptive_mean":
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 15
        )
    # default: adaptive_gaussian
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 15
    )


def enhance_document(
    image: np.ndarray,
    strength: int = 50,
    threshold_method: str = "adaptive_gaussian",
    remove_shadows: bool = True,
) -> np.ndarray:
    """Run the full enhancement pipeline on a (BGR or grayscale) document
    image and return a single-channel, enhanced result ready for display,
    download, and OCR.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

    gray = denoise(gray, strength=strength)
    if remove_shadows:
        gray = reduce_shadows(gray)
    gray = enhance_contrast(gray, strength=strength)
    result = apply_threshold(gray, method=threshold_method)
    return result
