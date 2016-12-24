"""
Tests for face.geometry module
"""

import shapely.geometry

import face.geometry

def test_get_bounding_box():

    left = 10
    upper = 20
    width = 5
    height = 10

    expected = shapely.geometry.Polygon([(10, 20), (15, 20), (15, 30), (10, 30)])
    actual = face.geometry.get_bounding_box(left, upper, width, height)

    assert expected.equals(actual)
