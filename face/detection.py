"""
Module with high level functionality for face detection
"""


class FaceCandidate:
    """
    A simple class representing an image crop that is to be examined for face presence.
    It contains three members:
    - image_coordinates that specify coordinates of the crop in image it was taken from
    - cropped_image - cropped image
    - focus_coordinates - coordinates within original image for which face prediction score should of the crop
    should be used. These are generally within image_coordinates, but not necessary the same, since many partially
    overlapping crops might be examined
    """

    def __init__(self, image_coordinates, cropped_image, focus_coordinates):
        """
        Constructor
        :param image_coordinates: specify coordinates of the crop in image it was taken from
        :param cropped_image: cropped image
        :param focus_coordinates: coordinates within original image for which face prediction score should of the crop
        should be used. These are generally within image_coordinates, but not necessary the same, since many partially
        overlapping crops might be examined
        """

        self.image_coordinates = image_coordinates
        self.cropped_image = cropped_image
        self.focus_coordinates = focus_coordinates
