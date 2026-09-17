import numpy as np

from app.utils.validation import (
    MAX_FILE_BYTES,
    MIN_HEIGHT,
    MIN_WIDTH,
    validate_file_extension,
    validate_file_size,
    validate_image_array,
)


def test_valid_extension():
    assert validate_file_extension("photo.jpg").is_valid
    assert validate_file_extension("photo.PNG").is_valid


def test_invalid_extension():
    result = validate_file_extension("document.pdf")
    assert not result.is_valid


def test_missing_extension():
    result = validate_file_extension("noext")
    assert not result.is_valid


def test_file_size_zero():
    assert not validate_file_size(0).is_valid


def test_file_size_too_large():
    assert not validate_file_size(MAX_FILE_BYTES + 1).is_valid


def test_file_size_ok():
    assert validate_file_size(1024).is_valid


def test_image_array_none():
    assert not validate_image_array(None).is_valid


def test_image_array_too_small():
    small = np.zeros((MIN_HEIGHT - 1, MIN_WIDTH - 1, 3), dtype=np.uint8)
    assert not validate_image_array(small).is_valid


def test_image_array_valid():
    ok = np.zeros((200, 200, 3), dtype=np.uint8)
    assert validate_image_array(ok).is_valid
