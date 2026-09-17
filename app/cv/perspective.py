"""
Perspective correction: given four ordered corner points, warp the image so
the document fills a flat, axis-aligned rectangle ("scanner" effect).
"""

from __future__ import annotations

import cv2
import numpy as np


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def compute_output_size(ordered_corners: np.ndarray) -> tuple:
    """Compute the target (width, height) for the warped output based on the
    longest observed edges, so the aspect ratio is preserved reasonably well
    even when the document was photographed at an angle.
    """
    (tl, tr, br, bl) = ordered_corners

    width_top = _distance(tr, tl)
    width_bottom = _distance(br, bl)
    max_width = max(int(width_top), int(width_bottom), 1)

    height_left = _distance(bl, tl)
    height_right = _distance(br, tr)
    max_height = max(int(height_left), int(height_right), 1)

    return max_width, max_height


def four_point_transform(image: np.ndarray, ordered_corners: np.ndarray) -> np.ndarray:
    """Warp ``image`` so the quadrilateral defined by ``ordered_corners``
    (TL, TR, BR, BL) becomes a flat rectangle.
    """
    max_width, max_height = compute_output_size(ordered_corners)

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(ordered_corners.astype("float32"), destination)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return warped
