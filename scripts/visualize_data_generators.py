"""
Module for visualizing outputs of data generators
"""

import vlogging

import face.utilities
import face.data_generators


def main():

    logger = face.utilities.get_logger()

    image_paths_file = "../../data/faces/small_dataset/training_image_paths.txt"
    bounding_boxes_file = "../../data/faces/small_dataset/training_bounding_boxes_list.txt"
    batch_size = 8

    generator = face.data_generators.get_batches_generator(image_paths_file, bounding_boxes_file, batch_size)

    for _ in range(4):

        batch = next(generator)

        batch = [image * 255 for image in batch]
        logger.info(vlogging.VisualRecord("Images batch", batch))


if __name__ == "__main__":

    main()