"""
Module with various image related processing functions
"""

import cv2


def scale_image_keeping_aspect_ratio(image, size):
    """
    Scale input image so that its smaller side becomes size large.
    Larger side is scaled so as to keep image aspect ration constant
    :param image: image
    :param size: size smaller side of rescaled image should have
    :return: rescaled image
    """

    smalled_dimension = image.shape[0] if image.shape[0] < image.shape[1] else image.shape[1]
    scale = size / smalled_dimension

    # Dimensions order has to be flipped for OpenCV
    target_shape = (int(scale * image.shape[1]), int(scale * image.shape[0]))
    return cv2.resize(image, target_shape)