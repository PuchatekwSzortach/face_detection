"""
Script for visualizing outputs of data generators
"""

import os

import vlogging
import numpy as np

import face.utilities
import face.data_generators
import face.processing
import face.models


def main():

    logger = face.utilities.get_logger()

    # dataset = "large_dataset"
    # dataset = "medium_dataset"
    dataset = "small_dataset"

    image_paths_file = os.path.join("../../data/faces/", dataset, "training_image_paths.txt")
    bounding_boxes_file = os.path.join("../../data/faces/", dataset, "training_bounding_boxes_list.txt")
    batch_size = 8

    generator = face.data_generators.get_batches_generator(image_paths_file, bounding_boxes_file, batch_size)

    images_count = face.utilities.get_file_lines_count(image_paths_file)

    image_shape = (224, 224, 3)
    model = face.models.get_pretrained_vgg_model(image_shape=image_shape)

    for _ in range(4):

        images, labels = next(generator)

        images = [image * 255 for image in images]
        images = [face.processing.scale_image_keeping_aspect_ratio(image, 100) for image in images]
        logger.info(vlogging.VisualRecord("Images batch", images, str(labels)))

        # predictions = model.predict(images)
        # print(predictions.shape)


if __name__ == "__main__":

    main()