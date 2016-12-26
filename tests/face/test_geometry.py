"""
Tests for face.geometry module
"""

import mock

import shapely.geometry

import face.geometry


def test_get_bounding_box():

    left = 10
    upper = 20
    width = 5
    height = 10

    expected = shapely.geometry.box(10, 20, 15, 30)
    actual = face.geometry.get_bounding_box(left, upper, width, height)

    assert expected.equals(actual)


def test_get_bounding_boxes_map():

    file_lines = "202599\nimage_id x_1 y_1 width height\n000001.jpg    95  71 226 313\n000002.jpg    72  94 221 306\n"
    mock_opener = mock.mock_open(read_data=file_lines)

    kwargs = {"open": mock_opener}

    first_bounding_box = shapely.geometry.box(95, 71, 95 + 226, 71 + 313)
    second_bounding_box = shapely.geometry.box(72, 94, 72 + 221, 94 + 306)

    expected = {
        "000001.jpg": first_bounding_box,
        "000002.jpg": second_bounding_box
    }

    actual = face.geometry.get_bounding_boxes_map("whatever", **kwargs)

    assert "000001.jpg" in expected
    assert "000002.jpg" in expected

    assert first_bounding_box.equals(actual["000001.jpg"])
    assert second_bounding_box.equals(actual["000002.jpg"])


def test_get_intersection_over_union_simple_intersection():

    first_polygon = shapely.geometry.box(10, 10, 20, 20)
    second_polygon = shapely.geometry.box(10, 10, 15, 15)

    assert 0.25 == face.geometry.get_intersection_over_union(first_polygon, second_polygon)
    assert 0.25 == face.geometry.get_intersection_over_union(second_polygon, first_polygon)


def test_get_intersection_over_union_non_intersecting_polygons():

    first_polygon = shapely.geometry.box(10, 10, 20, 20)
    second_polygon = shapely.geometry.box(100, 100, 150, 150)

    assert 0 == face.geometry.get_intersection_over_union(first_polygon, second_polygon)
    assert 0 == face.geometry.get_intersection_over_union(second_polygon, first_polygon)