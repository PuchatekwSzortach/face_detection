"""
Module with data generators and related functionality
"""

import face.utilities


def get_batches_generator(paths_file, bounding_boxes_file, batch_size):

    paths = [path.strip() for path in face.utilities.get_file_lines(paths_file)]

    index = 0

    while True:

        paths_batch = paths[index:index + batch_size] if index + batch_size < len(paths) else \
            paths[index:] + paths[:index - len(paths) + batch_size]

        images_batch = [face.utilities.get_image(path) for path in paths_batch]

        yield(images_batch)

        if index + batch_size < len(paths):

            index += batch_size

        else:

            index = 0

            # shuffle(paths)

