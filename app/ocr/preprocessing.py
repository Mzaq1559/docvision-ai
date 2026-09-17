"""
Preparing an already-enhanced document image for OCR. Tesseract tends to do
better on images with a minimum text height, so this upsamples small images
and makes sure we hand it a clean single-channel array.
"""

from __future__ import annotations

import cv2
import numpy as np

MIN_OCR_WIDTH = 900


def prepare_for_ocr(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    height, width = gray.shape[:2]
    if width < MIN_OCR_WIDTH:
        scale = MIN_OCR_WIDTH / float(width)
        gray = cv2.resize(
            gray, (MIN_OCR_WIDTH, int(height * scale)), interpolation=cv2.INTER_CUBIC
        )

    return gray
