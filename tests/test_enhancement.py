import numpy as np

from app.cv.enhancement import (
    apply_threshold,
    enhance_contrast,
    enhance_document,
    reduce_shadows,
)


def _sample_gray() -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.uniform(50, 200, (100, 100))).astype(np.uint8)


def test_enhance_contrast_preserves_shape_and_dtype():
    gray = _sample_gray()
    result = enhance_contrast(gray, strength=50)
    assert result.shape == gray.shape
    assert result.dtype == np.uint8


def test_reduce_shadows_preserves_shape():
    gray = _sample_gray()
    result = reduce_shadows(gray)
    assert result.shape == gray.shape
    assert result.dtype == np.uint8


def test_apply_threshold_binarizes_image():
    gray = _sample_gray()
    result = apply_threshold(gray, method="adaptive_gaussian")
    unique_values = set(np.unique(result).tolist())
    assert unique_values.issubset({0, 255})


def test_apply_threshold_none_is_identity():
    gray = _sample_gray()
    result = apply_threshold(gray, method="none")
    assert np.array_equal(result, gray)


def test_apply_threshold_invalid_method_falls_back():
    gray = _sample_gray()
    result = apply_threshold(gray, method="not_a_real_method")
    unique_values = set(np.unique(result).tolist())
    assert unique_values.issubset({0, 255})


def test_enhance_document_end_to_end_bgr_input():
    bgr = np.zeros((120, 120, 3), dtype=np.uint8)
    bgr[:, :] = (180, 180, 180)
    result = enhance_document(bgr, strength=50, threshold_method="adaptive_gaussian")
    assert result.ndim == 2
    assert result.shape == (120, 120)
