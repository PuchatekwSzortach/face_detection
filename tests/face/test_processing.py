"""
Tests for face.processing module
"""

import numpy as np

import face.processing


def test_scale_image_keeping_aspect_ratio_vertial_image():

    image = np.zeros(shape=[10, 20])
    target_size = 30

    # Vertical dimension smaller
    rescaled_image = face.processing.scale_image_keeping_aspect_ratio(image, target_size)

    assert (30, 60) == rescaled_image.shape


def test_scale_image_keeping_aspect_ratio_horizontal_image():

    image = np.zeros(shape=[10, 5])
    target_size = 20

    # Vertical dimension smaller
    rescaled_image = face.processing.scale_image_keeping_aspect_ratio(image, target_size)

    assert (40, 20) == rescaled_image.shape
