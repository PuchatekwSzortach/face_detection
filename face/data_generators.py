"""
Module with data generators and related functionality
"""

import face.utilities
import face.geometry
import face.processing


def get_batches_generator(paths_file, bounding_boxes_file, batch_size):

    if batch_size % 4 != 0:

        raise ValueError("Batch size must be divisible by 4!")

    paths = [path.strip() for path in face.utilities.get_file_lines(paths_file)]
    print("No paths shuffling yet")

    bounding_boxes_map = face.geometry.get_bounding_boxes_map(bounding_boxes_file)

    index = 0

    while True:

        batch = face.processing.get_data_batch(paths, bounding_boxes_map, index, batch_size)
        yield(batch)

        if index + batch_size < len(paths):

            index += batch_size

        else:

            index = 0

            print("No shuffling added!")
            # shuffle(paths)

