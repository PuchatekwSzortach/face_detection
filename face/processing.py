"""
Module with various image related processing functions
"""

import cv2

import face.utilities


class InvalidBoundingBoxError(Exception):
    """
    A simple exception used when bounding boxes appear invalid and hence faces crop can't be taken
    """

    pass


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


def get_data_batch(paths, bounding_boxes_map, index, batch_size):

    images_batch = []

    while len(images_batch) < batch_size:

        image = face.utilities.get_image(paths[index])
        images_batch.append(image)

        index += 1

        if index >= len(paths):

            index = 0

    return images_batch
