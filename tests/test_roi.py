import numpy as np
import pytest

from pimpsi.roi import Roi


def test_rectangle_mask_uses_image_shape_height_width():
    roi = Roi(
        id="roi_001",
        label="box",
        shape_type="rectangle",
        vertices_xy=[(1.0, 1.0), (4.0, 3.0)],
    )

    mask = roi.to_mask((4, 5))

    expected = np.zeros((4, 5), dtype=bool)
    expected[1:3, 1:4] = True
    np.testing.assert_array_equal(mask, expected)


def test_roi_polygon_mask():
    roi = Roi(
        id="roi_001",
        label="polygon",
        shape_type="polygon",
        vertices_xy=[(1.0, 1.0), (4.0, 1.0), (4.0, 4.0), (1.0, 4.0)],
    )

    mask = roi.to_mask((5, 5))

    expected = np.zeros((5, 5), dtype=bool)
    expected[1:4, 1:4] = True
    np.testing.assert_array_equal(mask, expected)


def test_roi_roundtrip_preserves_geometry():
    roi = Roi(
        id="roi_001",
        label="ellipse",
        shape_type="ellipse",
        vertices_xy=[(0.5, 1.5), (5.5, 6.5)],
        visible=False,
        locked=True,
        group="treated",
        notes="keep",
    )

    assert Roi.from_dict(roi.to_dict()) == roi


def test_polygon_requires_three_vertices():
    roi = Roi(id="bad", label="bad", shape_type="polygon", vertices_xy=[(0.0, 0.0), (1.0, 1.0)])

    with pytest.raises(ValueError, match="at least three"):
        roi.to_mask((4, 4))

