"""
Module with high level functionality for face detection
"""

import shapely.geometry


class FaceCandidate:
    """
    A simple class representing an image crop that is to be examined for face presence.
    It contains three members:
    - crop_coordinates that specify coordinates of the crop in image it was taken from
    - cropped_image - cropped image
    - focus_coordinates - coordinates within original image for which face prediction score should of the crop
    should be used. These are generally within image_coordinates, but not necessary the same, since many partially
    overlapping crops might be examined
    """

    def __init__(self, crop_coordinates, cropped_image, focus_coordinates):
        """
        Constructor
        :param crop_coordinates: specify coordinates of the crop in image it was taken from
        :param cropped_image: cropped image
        :param focus_coordinates: coordinates within original image for which face prediction score should of the crop
        should be used. These are generally within image_coordinates, but not necessary the same, since many partially
        overlapping crops might be examined
        """

        self.crop_coordinates = crop_coordinates
        self.cropped_image = cropped_image
        self.focus_coordinates = focus_coordinates


def get_face_candidates(image, crop_size, step):
    """
    Given an image, crop size and step, get list of face candidates - crops of input image
    that will be examined for face presence. Each crop is of crop_size and crops are taken at step distance from
     upper left corner of one crop to next crop (thus crops might be overlapping if step is smaller than crop_size).
     Once all possible crops have been taken scanning image in one row, scanning proceeds from first column of
     row step away from current row.
    :param image: image from which crops are to be taken
    :param crop_size: size of each crop
    :param step: step at which crops should be taken. Must be not larger than crop size.
    :return: list of FaceCandidate objects
    """

    if crop_size < step:

        raise ValueError("Crop size ({}) must be not smaller than step size ({})".format(crop_size, step))

    face_candidates = []

    y = 0

    while y + crop_size <= image.shape[0]:

        x = 0

        while x + crop_size <= image.shape[1]:

            crop_coordinates = shapely.geometry.box(x, y, x + crop_size, y + crop_size)
            cropped_image = image[y:y + crop_size, x:x + crop_size]
            focus_coordinates = shapely.geometry.box(x, y, x + step, y + step)

            candidate = FaceCandidate(crop_coordinates, cropped_image, focus_coordinates)
            face_candidates.append(candidate)

            x += step

        y += step

    return face_candidates
