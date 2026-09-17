import cv2
import numpy as np

from app.cv.detector import detect_document, order_points


def _make_document_photo(rotated: bool = False) -> np.ndarray:
    """Synthetic photo: a white rectangle (the "document") on a dark
    background, optionally rotated, so the detector has real edges to find.
    """
    canvas = np.full((500, 500, 3), 40, dtype=np.uint8)
    doc = np.full((500, 500, 3), 40, dtype=np.uint8)
    cv2.rectangle(doc, (100, 100), (400, 400), (255, 255, 255), -1)

    if rotated:
        matrix = cv2.getRotationMatrix2D((250, 250), 15, 1.0)
        doc = cv2.warpAffine(doc, matrix, (500, 500), borderValue=(40, 40, 40))

    return doc


def test_order_points_returns_tl_tr_br_bl():
    pts = np.array([[10, 10], [10, 110], [110, 110], [110, 10]], dtype="float32")
    ordered = order_points(pts)
    tl, tr, br, bl = ordered
    assert tl[0] < tr[0] and tl[1] < bl[1]
    assert br[0] > bl[0] and br[1] > tr[1]


def test_detect_document_on_clean_rectangle():
    image = _make_document_photo(rotated=False)
    result = detect_document(image, sensitivity=50)
    assert result.corners.shape == (4, 2)
    # Corners should be roughly within the drawn rectangle bounds.
    xs = result.corners[:, 0]
    ys = result.corners[:, 1]
    assert xs.min() >= 60 and xs.max() <= 440
    assert ys.min() >= 60 and ys.max() <= 440


def test_detect_document_on_rotated_rectangle():
    image = _make_document_photo(rotated=True)
    result = detect_document(image, sensitivity=50)
    assert result.corners.shape == (4, 2)


def test_detect_document_fallback_on_blank_image():
    blank = np.full((300, 300, 3), 128, dtype=np.uint8)
    result = detect_document(blank, sensitivity=50)
    assert result.corners.shape == (4, 2)
    assert result.used_fallback is True


def test_detect_document_raises_on_empty_input():
    import pytest

    with pytest.raises(ValueError):
        detect_document(np.zeros((0, 0, 3), dtype=np.uint8))
