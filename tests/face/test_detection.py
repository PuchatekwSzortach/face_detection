"""
Tests for face.detection module
"""

import mock

import numpy as np
import shapely.geometry
import pytest

import face.detection


def test_test_get_face_candidates_check_raises_on_step_larger_than_crop_size():

    with pytest.raises(ValueError):

        face.detection.get_face_candidates(np.zeros(shape=[10, 10]), crop_size=4, step=5)


def test_get_face_candidates_single_row_crops():

    image = np.arange(40).reshape([4, 10])
    crop_size = 4
    step = 3

    face_candidates = face.detection.get_face_candidates(image, crop_size, step)

    assert 3 == len(face_candidates)

    # Assert properties of first candidate
    assert 0 == np.min(face_candidates[0].cropped_image)
    assert 33 == np.max(face_candidates[0].cropped_image)

    assert shapely.geometry.box(0, 0, 4, 4) == face_candidates[0].crop_coordinates
    assert shapely.geometry.box(0, 0, 3, 3) == face_candidates[0].focus_coordinates

    # Assert properties of second candidate
    assert 3 == np.min(face_candidates[1].cropped_image)
    assert 36 == np.max(face_candidates[1].cropped_image)

    assert shapely.geometry.box(3, 0, 7, 4) == face_candidates[1].crop_coordinates
    assert shapely.geometry.box(3, 0, 6, 3) == face_candidates[1].focus_coordinates

    # Assert properties of third candidate
    assert 6 == np.min(face_candidates[2].cropped_image)
    assert 39 == np.max(face_candidates[2].cropped_image)

    assert shapely.geometry.box(6, 0, 10, 4) == face_candidates[2].crop_coordinates
    assert shapely.geometry.box(6, 0, 9, 3) == face_candidates[2].focus_coordinates


def test_get_face_candidates_single_column_crops():

    image = np.arange(75).reshape([15, 5])
    crop_size = 5
    step = 4

    face_candidates = face.detection.get_face_candidates(image, crop_size, step)

    assert 3 == len(face_candidates)

    # Assert properties of first candidate
    assert 0 == np.min(face_candidates[0].cropped_image)
    assert 24 == np.max(face_candidates[0].cropped_image)

    assert shapely.geometry.box(0, 0, 5, 5) == face_candidates[0].crop_coordinates
    assert shapely.geometry.box(0, 0, 4, 4) == face_candidates[0].focus_coordinates

    # Assert properties of second candidate
    assert 20 == np.min(face_candidates[1].cropped_image)
    assert 44 == np.max(face_candidates[1].cropped_image)

    assert shapely.geometry.box(0, 4, 5, 9) == face_candidates[1].crop_coordinates
    assert shapely.geometry.box(0, 4, 4, 8) == face_candidates[1].focus_coordinates

    # Assert properties of third candidate
    assert 40 == np.min(face_candidates[2].cropped_image)
    assert 64 == np.max(face_candidates[2].cropped_image)

    assert shapely.geometry.box(0, 8, 5, 13) == face_candidates[2].crop_coordinates
    assert shapely.geometry.box(0, 8, 4, 12) == face_candidates[2].focus_coordinates


def test_get_face_candidates_simple_grid():

    image = np.arange(100).reshape([10, 10])
    crop_size = 5
    step = 4

    face_candidates = face.detection.get_face_candidates(image, crop_size, step)

    assert 4 == len(face_candidates)

    # Assert properties of first candidate
    assert 0 == np.min(face_candidates[0].cropped_image)
    assert 44 == np.max(face_candidates[0].cropped_image)

    assert shapely.geometry.box(0, 0, 5, 5) == face_candidates[0].crop_coordinates
    assert shapely.geometry.box(0, 0, 4, 4) == face_candidates[0].focus_coordinates

    # Assert properties of second candidate
    assert 4 == np.min(face_candidates[1].cropped_image)
    assert 48 == np.max(face_candidates[1].cropped_image)

    assert shapely.geometry.box(4, 0, 9, 5) == face_candidates[1].crop_coordinates
    assert shapely.geometry.box(4, 0, 8, 4) == face_candidates[1].focus_coordinates

    # Assert properties of third candidate
    assert 40 == np.min(face_candidates[2].cropped_image)
    assert 84 == np.max(face_candidates[2].cropped_image)

    assert shapely.geometry.box(0, 4, 5, 9) == face_candidates[2].crop_coordinates
    assert shapely.geometry.box(0, 4, 4, 8) == face_candidates[2].focus_coordinates

    # Assert properties of fourth candidate
    assert 44 == np.min(face_candidates[3].cropped_image)
    assert 88 == np.max(face_candidates[3].cropped_image)

    assert shapely.geometry.box(4, 4, 9, 9) == face_candidates[3].crop_coordinates
    assert shapely.geometry.box(4, 4, 8, 8) == face_candidates[3].focus_coordinates

#
# def test_get_heatmap_simple():
#
#     image = np.zeros(shape=[10, 10])
#
#     mock_model = mock.Mock()
#     mock_model.predict.return_value = [0.2, 0.4, 0.6, 0.8]
#
#     crop_size = 5
#     step = 4
#
#     expected_heatmap = np.zeros(shape=[10, 10])
#     expected_heatmap[:4, :4] = 0.2
#     expected_heatmap[:4, 4:8] = 0.4
#     expected_heatmap[4:8, :4] = 0.6
#     expected_heatmap[4:8, 4:8] = 0.8
#
#     computer = face.detection.HeatmapComputer(image, mock_model, crop_size, step)
#     actual_heatmap = computer.get_heatmap()
#
#     # assert np.all(expected_heatmap == actual_heatmap)
