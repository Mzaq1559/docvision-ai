import numpy as np

from app.cv.perspective import compute_output_size, four_point_transform


def test_compute_output_size_axis_aligned_square():
    corners = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype="float32")
    width, height = compute_output_size(corners)
    assert width == 100
    assert height == 100


def test_four_point_transform_output_shape():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    image[50:150, 50:150] = 255  # white square "document"
    corners = np.array([[50, 50], [150, 50], [150, 150], [50, 150]], dtype="float32")
    warped = four_point_transform(image, corners)
    assert warped.shape[0] == 100
    assert warped.shape[1] == 100
    # The warped result should be (close to) solid white since we warped
    # exactly the white region.
    assert warped.mean() > 200


def test_four_point_transform_handles_skewed_quad():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    corners = np.array([[20, 30], [180, 10], [190, 190], [10, 170]], dtype="float32")
    warped = four_point_transform(image, corners)
    assert warped.shape[0] > 0 and warped.shape[1] > 0
