"""
Script with a bunch useful visualization functions like showing output of data generators, crops predictions,
 heatmaps, etc. Useful both for debugging and quick demos.
"""

import os

import vlogging
import numpy as np

import face.utilities
import face.data_generators
import face.processing
import face.models


def log_data_batches(data_generator, logger):

    for _ in range(4):

        images, labels = next(data_generator)

        images = [image * 255 for image in images]
        images = [face.processing.scale_image_keeping_aspect_ratio(image, 100) for image in images]
        logger.info(vlogging.VisualRecord("Images batch", images, str(labels)))


def log_crops_predictions(data_generator, logger):

    image_shape = (224, 224, 3)
    model = face.models.get_pretrained_vgg_model(image_shape=image_shape)
    model.load_weights("../../data/faces/models/model.h5")

    for _ in range(4):

        images, _ = next(data_generator)
        predictions = model.predict(images)

        images = [image * 255 for image in images]
        images = [face.processing.scale_image_keeping_aspect_ratio(image, 100) for image in images]

        logger.info(vlogging.VisualRecord("Crops predictions", images, str(predictions)))


def main():

    logger = face.utilities.get_logger()

    # dataset = "large_dataset"
    # dataset = "medium_dataset"
    dataset = "small_dataset"

    image_paths_file = os.path.join("../../data/faces/", dataset, "training_image_paths.txt")
    bounding_boxes_file = os.path.join("../../data/faces/", dataset, "training_bounding_boxes_list.txt")
    batch_size = 8

    generator = face.data_generators.get_batches_generator(image_paths_file, bounding_boxes_file, batch_size)

    # log_data_batches(generator, logger)
    log_crops_predictions(generator, logger)


if __name__ == "__main__":

    main()