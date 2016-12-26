"""
Module with geometry related functions, mostly relating to bounding boxes processing
"""

import shapely.geometry


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

