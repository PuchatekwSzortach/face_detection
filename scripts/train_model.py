"""
Script for training models
"""

import os

import keras

import face.utilities
import face.models
import face.data_generators
import face.config


def main():

    logger = face.utilities.get_logger()

    # dataset = "large_dataset"
    dataset = "medium_dataset"
    # dataset = "small_dataset"

    training_image_paths_file = os.path.join("../../data/faces/", dataset, "training_image_paths.txt")
    training_bounding_boxes_file = os.path.join("../../data/faces/", dataset, "training_bounding_boxes_list.txt")

    validation_image_paths_file = os.path.join("../../data/faces/", dataset, "validation_image_paths.txt")
    validation_bounding_boxes_file = os.path.join("../../data/faces/", dataset, "validation_bounding_boxes_list.txt")
    batch_size = face.config.batch_size

    model_path = face.config.model_path
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=model_path, save_best_only=True)

    model = face.models.get_pretrained_vgg_model(image_shape=face.config.image_shape)

    training_data_generator = face.data_generators.get_batches_generator(
        training_image_paths_file, training_bounding_boxes_file, batch_size)

    validation_data_generator = face.data_generators.get_batches_generator(
        validation_image_paths_file, validation_bounding_boxes_file, batch_size)

    model.fit_generator(
        training_data_generator, samples_per_epoch=face.utilities.get_file_lines_count(training_image_paths_file),
        nb_epoch=10,
        validation_data=validation_data_generator,
        nb_val_samples=face.utilities.get_file_lines_count(validation_image_paths_file),
        callbacks=[model_checkpoint]
    )


if __name__ == "__main__":

    main()
