"""
Input validation helpers.

These are intentionally defensive: anything the user can upload should be
checked here before it ever reaches the CV pipeline, so the rest of the
codebase can assume it is working with a valid BGR ``numpy.ndarray``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

SUPPORTED_EXTENSIONS = ("jpg", "jpeg", "png", "webp")

MIN_WIDTH = 64
MIN_HEIGHT = 64
MAX_FILE_BYTES = 25 * 1024 * 1024  # 25 MB


@dataclass
class ValidationResult:
    is_valid: bool
    message: str = ""


def validate_file_extension(filename: str) -> ValidationResult:
    if not filename or "." not in filename:
        return ValidationResult(False, "The file has no recognizable extension.")
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return ValidationResult(
            False,
            f"Unsupported file format '.{ext}'. Please upload one of: "
            f"{', '.join(e.upper() for e in SUPPORTED_EXTENSIONS)}.",
        )
    return ValidationResult(True)


def validate_file_size(num_bytes: int) -> ValidationResult:
    if num_bytes <= 0:
        return ValidationResult(False, "The uploaded file appears to be empty.")
    if num_bytes > MAX_FILE_BYTES:
        return ValidationResult(
            False,
            f"The file is too large ({num_bytes / (1024 * 1024):.1f} MB). "
            f"Please upload an image under {MAX_FILE_BYTES // (1024 * 1024)} MB.",
        )
    return ValidationResult(True)


def validate_image_array(image: Optional[np.ndarray]) -> ValidationResult:
    """Validate a decoded image array before it enters the CV pipeline."""
    if image is None:
        return ValidationResult(
            False,
            "The file could not be decoded as an image. It may be corrupted "
            "or in an unsupported format.",
        )
    if not isinstance(image, np.ndarray):
        return ValidationResult(False, "Decoded content is not a valid image array.")
    if image.ndim not in (2, 3):
        return ValidationResult(False, "The image has an unexpected number of dimensions.")

    height, width = image.shape[:2]
    if width < MIN_WIDTH or height < MIN_HEIGHT:
        return ValidationResult(
            False,
            f"The image is too small ({width}x{height}px). Please upload an "
            f"image at least {MIN_WIDTH}x{MIN_HEIGHT}px.",
        )
    if image.size == 0:
        return ValidationResult(False, "The image contains no pixel data.")
    return ValidationResult(True)
