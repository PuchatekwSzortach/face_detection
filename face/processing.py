"""
Module with various image related processing functions
"""

import os
import random

import numpy as np
import cv2
import shapely.geometry

import face.utilities
import face.geometry
import face.config


class InvalidBoundingBoxError(Exception):
    """
    A simple exception used when bounding boxes appear invalid and hence faces crop can't be taken
    """

    pass


class CropException(Exception):
    """
    A simple exception used when no good image crop could be obtained
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


def get_data_batch(paths, bounding_boxes_map, index, batch_size, crop_size):
    """
    Create a single batch of face and non-face crops, along with labels.
    :param paths: list of image paths
    :param bounding_boxes_map: {path: face bounding box} dictionary
    :param index: index from which paths list should be read
    :param batch_size: size of batch to be returned
    :param crop_size: size of each image crop in batch
    :return: tuple (image_crops, labels)
    """

    images_batch = []
    labels_batch = []

    while len(images_batch) < batch_size and len(labels_batch) < batch_size:

        try:

            path = paths[index]
            image = face.utilities.get_image(path)

            image_bounding_box = shapely.geometry.box(0, 0, image.shape[1], image.shape[0])
            face_bounding_box = bounding_boxes_map[os.path.basename(path)]

            # Only allow images for which face covers at least 1% of the image. If it doesn't, then face bounding
            # box is probably incorrect
            if face.geometry.get_intersection_over_union(image_bounding_box, face_bounding_box) < 0.01:

                raise InvalidBoundingBoxError("Invalid bounding box for image {}".format(path))

            scale = face.geometry.get_scale(face_bounding_box, crop_size)

            scaled_image = get_scaled_image(image, scale)
            scaled_bounding_box = face.geometry.get_scaled_bounding_box(face_bounding_box, scale)

            # Randomly flip image
            if random.randint(0, 1) == 1:
                
                scaled_image = cv2.flip(scaled_image, flipCode=1)

                scaled_bounding_box = face.geometry.flip_bounding_box_about_vertical_axis(
                    scaled_bounding_box, scaled_image.shape)

            crops, labels = get_image_crops_labels_batch(scaled_image, scaled_bounding_box, crop_size=crop_size)

            images_batch.extend(crops)
            labels_batch.extend(labels)

        # If image had an invalid bounding box, we want to skip over that image and go to next one
        except (InvalidBoundingBoxError, CropException):

            pass

        index += 1

        if index >= len(paths):

            index = 0

    # Shuffle within a batch
    batch = list(zip(images_batch, labels_batch))
    random.shuffle(batch)
    images_batch, labels_batch = zip(*batch)

    return np.array(images_batch), np.array(labels_batch)


def get_scaled_image(image, scale):
    """
    Scales image. A thin wrapper around cv2.resize that makes sure that resulting image
    maps to integer sizes
    :param image: image
    :param scale: float
    :return: scaled image
    """

    return cv2.resize(image, (int(scale * image.shape[1]), int(scale * image.shape[0])))


def get_image_crops_labels_batch(image, face_bounding_box, crop_size):
    """
    Given an image and a bounding box of face in it, return a tuple (crops, labels), where both crops and labels
    are a list of length 4. Crops contain random image crops of size crop_size x crop_size such that
    one crop has a high IOU with face in the picture, while remaining crops have low IOU with face.
    Corresponding labels are 1 for face and 0 for lack of face. Function throws CropException when no good
    crops could be obtained within a reasonable amount of attempts.
    :param image: image
    :param face_bounding_box: bounding box of face in image
    :param crop_size: desired crop size
    :return: (crops, labels) tuple
    """

    face_crop = get_random_face_crop(image, face_bounding_box, crop_size)
    non_face_crops = [get_random_non_face_crop(image, face_bounding_box, crop_size) for _ in range(3)]

    crops = [face_crop] + non_face_crops
    labels = [1, 0, 0, 0]

    return crops, labels


def get_random_face_crop(image, face_bounding_box, crop_size):
    """
    Given an image and face bounding box, return a random crop that has high IOU with face bounding box and
    is of size crop_size x crop_size
    :param image: image
    :param face_bounding_box: bounding box of face
    :param crop_size: desired crop size
    :return: random crop that mostly contains face
    """

    bounds = face_bounding_box.bounds

    # Try up to x times to get a good crop
    for index in range(100):

        x = int(bounds[0]) + random.randint(-crop_size, crop_size)
        y = int(bounds[1]) + random.randint(-crop_size, crop_size)

        cropped_region = shapely.geometry.box(x, y, x + crop_size, y + crop_size)

        are_coordinates_legal = x >= 0 and y >= 0 and \
                                x + crop_size < image.shape[1] and y + crop_size < image.shape[0]

        is_iou_high = face.geometry.get_intersection_over_union(face_bounding_box, cropped_region) > 0.5

        if are_coordinates_legal and is_iou_high:

            return image[y:y + crop_size, x:x + crop_size]

    # We failed to find a good crop despite trying x times, throw
    raise CropException()


def get_random_non_face_crop(image, face_bounding_box, crop_size):
    """
    Given an image and face bounding box, return a random crop that has low IOU with face bounding box and
    is of size crop_size x crop_size
    :param image: image
    :param face_bounding_box: bounding box of face
    :param crop_size: desired crop size
    :return: random crop that contains little or no face
    """

    # Try up to x times to get a good crop
    for index in range(100):

        x = random.randint(0, image.shape[1] - crop_size)
        y = random.randint(0, image.shape[0] - crop_size)

        cropped_region = shapely.geometry.box(x, y, x + crop_size, y + crop_size)

        are_coordinates_legal = x >= 0 and y >= 0 and \
                                x + crop_size < image.shape[1] and y + crop_size < image.shape[0]

        is_iou_low = face.geometry.get_intersection_over_union(face_bounding_box, cropped_region) < 0.5

        if are_coordinates_legal and is_iou_low:

            return image[y:y + crop_size, x:x + crop_size]

    # We failed to find a good crop despite trying x times, throw
    raise CropException()


def get_smallest_expected_face_size(image_shape, min_face_size, min_face_to_image_ratio):
    """
    Given an image shape, minimum face size and minimum face to image ratio, compute smallest
    expected face size for the image.
    :param image_shape: tuple of integers
    :param min_face_size: integer
    :param min_face_to_image_ratio: float
    :return: integer, larger of min_face_size and min(image_shape) * min_face_to_image_ratio
    """

    image_ratio_based_size = int(min(image_shape) * min_face_to_image_ratio)
    return max(min_face_size, image_ratio_based_size)
