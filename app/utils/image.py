"""
Shared image I/O and resizing helpers used across the CV pipeline.

Keeping these in one place makes it easy to reason about CPU cost: every
resize decision and every color-space conversion funnels through here.
"""

from __future__ import annotations

import io
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode raw image bytes (as received from a Streamlit uploader) into a
    BGR ``numpy.ndarray``, the convention OpenCV expects. Returns ``None``
    (well, raises) rather than silently producing garbage on bad input.
    """
    file_bytes = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        # Fall back to Pillow, which handles a couple of formats/edge cases
        # (e.g. some WEBP variants) that cv2.imdecode can be picky about.
        try:
            pil_image = Image.open(io.BytesIO(data)).convert("RGB")
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception:
            return None
    return image


def encode_image_to_bytes(image: np.ndarray, ext: str = ".png") -> bytes:
    """Encode a BGR ``numpy.ndarray`` back to bytes for download buttons."""
    success, buffer = cv2.imencode(ext, image)
    if not success:
        raise ValueError("Failed to encode image.")
    return buffer.tobytes()


def resize_max_width(image: np.ndarray, max_width: int) -> Tuple[np.ndarray, float]:
    """Downscale ``image`` so its width does not exceed ``max_width``.

    This is the single most important CPU optimization in the app: every
    expensive step (Canny, contour search, denoising) runs on the resized
    image rather than the original, often multi-megapixel, photo.

    Returns the resized image and the scale factor applied (<= 1.0), so
    callers can map coordinates back to the original resolution if needed.
    """
    height, width = image.shape[:2]
    if width <= max_width or max_width <= 0:
        return image, 1.0
    scale = max_width / float(width)
    new_size = (max_width, max(1, int(round(height * scale))))
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized, scale


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert BGR (OpenCV convention) to RGB (Streamlit/PIL convention)."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
