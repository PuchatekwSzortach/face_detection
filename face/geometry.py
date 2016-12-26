"""
Module with geometry related functions, mostly relating to bounding boxes processing
"""

import shapely.geometry
import shapely.affinity


def get_bounding_box(left, top, width, height):
    """
    Given left and top coordinates and width and height of a bounding box, return a bounding box instance
    :param left: x-coordinate of left side
    :param top: y-coordinate of top side
    :param width: width
    :param height: height
    :return: shapely.geometry.Polygon instance representing a bounding box
    """

    return shapely.geometry.box(left, top, left + width, top + height)


def get_bounding_boxes_map(path, **kwargs):
    """
    Give a path to a file with bounding boxes for images, return a dictionary mapping image names to
    bounding boxes
    :param path: path to file
    :return: {image file name: shapely.geometry.Polygon} dictionary
    """

    file_opener = kwargs["open"] if "open" in kwargs.keys() else open

    map = {}

    with file_opener(path) as file:

        # Discard first 2 lines, as they represent header
        data = file.readlines()[2:]

        for line in data:

            tokens = line.split()

            file_name = tokens[0]
            int_tokens = [int(token) for token in tokens[1:]]

            map[file_name] = get_bounding_box(*int_tokens)

    return map


def get_intersection_over_union(first_polygon, second_polygon):
    """
    Given two polygons returns their intersection over union
    :param first_polygon: shapely.geometry.Polygon instance
    :param second_polygon: shapely.geometry.Polygon instance
    :return: float
    """

    intersection_polygon = first_polygon.intersection(second_polygon)
    union_polygon = first_polygon.union(second_polygon)

    return intersection_polygon.area / union_polygon.area


def get_scale(bounding_box, target_size):
    """
    Get a scale that would bring smaller side of bounding box to have target_size
    :param bounding_box: bounding box
    :param target_size: target size for smaller bounding box side
    :return: float
    """

    horizontal_side = bounding_box.bounds[2] - bounding_box.bounds[0]
    vertical_side = bounding_box.bounds[3] - bounding_box.bounds[1]

    smaller_side = horizontal_side if horizontal_side < vertical_side else vertical_side

    return target_size / smaller_side


def get_scaled_bounding_box(bounding_box, scale):
    """
    Given a bounding box and a scale, return scaled bounding box. Note that scaling is done w.r.t. axis origin,
    hence this operation can change all bounding boxes coordinates
    :param bounding_box: bounding box
    :param scale: scale
    :return: rescaled bounding box
    """

    return shapely.affinity.affine_transform(bounding_box, [scale, 0, 0, scale, 0, 0])


def flip_bounding_box_about_vertical_axis(bounding_box, image_shape):
    """
    Given a bounding box and image shape, flip the box about vertical axis of the image
    :param bounding_box: bounding box
    :param image_shape: image shape
    :return: flipped bounding box
    """

    bounds = bounding_box.bounds
    return shapely.geometry.box(image_shape[1] - bounds[0], bounds[1], image_shape[1] - bounds[2], bounds[3])


