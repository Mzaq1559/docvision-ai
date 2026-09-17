"""
Lightweight metrics helpers. No fake numbers: every value shown in the
Streamlit "Processing Analytics" panel is measured from a real run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


class Timer:
    """Context manager / manual stopwatch for measuring wall-clock time.

    Usage:
        timer = Timer()
        with timer:
            do_work()
        print(timer.elapsed_seconds)
    """

    def __init__(self) -> None:
        self._start: Optional[float] = None
        self.elapsed_seconds: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._start is not None:
            self.elapsed_seconds = time.perf_counter() - self._start


@dataclass
class ProcessingStats:
    original_resolution: tuple = (0, 0)
    processed_resolution: tuple = (0, 0)
    processing_time_seconds: float = 0.0
    document_area_ratio: float = 0.0
    ocr_character_count: int = 0
    fields_detected: int = 0

    def as_display_dict(self) -> dict:
        ow, oh = self.original_resolution
        pw, ph = self.processed_resolution
        return {
            "Processing Time": f"{self.processing_time_seconds:.2f} sec",
            "Original Resolution": f"{ow} × {oh}",
            "Processed Resolution": f"{pw} × {ph}",
            "Document Area": f"{self.document_area_ratio * 100:.1f}%",
            "OCR Characters": str(self.ocr_character_count),
            "Fields Detected": str(self.fields_detected),
        }


def document_area_ratio(image_shape: tuple, contour: Optional[np.ndarray]) -> float:
    """Fraction of the full image area covered by the detected document contour."""
    if contour is None:
        return 0.0
    height, width = image_shape[:2]
    total_area = float(height * width)
    if total_area <= 0:
        return 0.0
    doc_area = float(abs(cv2.contourArea(contour)))
    return max(0.0, min(1.0, doc_area / total_area))
