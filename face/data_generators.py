"""
Module with data generators and related functionality
"""

import random

import face.utilities
import face.geometry
import face.processing


def get_batches_generator(paths_file, bounding_boxes_file, batch_size, crop_size):
    """
    Returns a generator that produces batches of face and non-face image crops, along with labels.
    A single image is cut into four random crops, with one containing face and remaining 3 without it.
    :param paths_file: path to file with image paths
    :param bounding_boxes_file: path to file with bounding boxes of faces in each image
    :param batch_size: size of a single batch to be outputted by generator
    :param crop_size: size image crops should have
    :return: batches generator
    """

    if batch_size % 4 != 0:

        raise ValueError("Batch size must be divisible by 4!")

    images_per_batch = batch_size // 4

    paths = [path.strip() for path in face.utilities.get_file_lines(paths_file)]
    random.shuffle(paths)

    bounding_boxes_map = face.geometry.get_bounding_boxes_map(bounding_boxes_file)

    index = 0

    while True:

        batch = face.processing.get_data_batch(paths, bounding_boxes_map, index, batch_size, crop_size)
        yield(batch)

        if index + images_per_batch < len(paths):

            index += images_per_batch

        else:

            index = 0
            random.shuffle(paths)

