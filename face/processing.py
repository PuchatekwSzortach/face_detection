"""
Module with various image related processing functions
"""

import os

import cv2
import shapely.geometry

import face.utilities
import face.geometry


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

        try:

            path = paths[index]
            image = face.utilities.get_image(path)

            image_bounding_box = shapely.geometry.Polygon(
                [(0, 0), (image.shape[1], 0), (image.shape[1], image.shape[0]), (0, image.shape[0])])

            face_bounding_box = bounding_boxes_map[os.path.basename(path)]

            # Only allow images for which face covers at least 1% of the image. If it doesn't, then face bounding
            # box is probably incorrect
            if face.geometry.get_intersection_over_union(image_bounding_box, face_bounding_box) < 0.01:

                raise InvalidBoundingBoxError("Invalid bounding box for image {}".format(path))

            images_batch.append(image)

        # If image had an invalid bounding box, we want to skip over that image and go to next one
        except InvalidBoundingBoxError:

            print(paths[index])
            pass

        index += 1

        if index >= len(paths):

            index = 0

    return images_batch
