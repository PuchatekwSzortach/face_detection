"""
Module with high level functionality for face detection
"""

import shapely.geometry
import numpy as np
import cv2

import face.utilities
import face.geometry
import face.processing


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


class FaceDetection:
    """
    A very simple class representing a face detection. Contains face bounding box and detection score.
    """

    def __init__(self, bounding_box, score):
        """
        Constructor
        :param bounding_box: bounding box of the face
        :param score: confidence score of detection
        """

        self.bounding_box = bounding_box
        self.score = score

    def __eq__(self, other):
        """
        Equality comparison
        :param other: object to compare to
        :return: boolean value
        """

        if not isinstance(other, self.__class__):

            return False

        return self.bounding_box.equals(other.bounding_box) and self.score == other.score


def get_face_candidates(image, crop_size, stride):
    """
    Given an image, crop size and stride, get list of face candidates - crops of input image
    that will be examined for face presence. Each crop is of crop_size and crops are taken at stride distance from
     upper left corner of one crop to next crop (thus crops might be overlapping if stride is smaller than crop_size).
     Once all possible crops have been taken scanning image in one row, scanning proceeds from first column of
     row stride away from current row.
    :param image: image from which crops are to be taken
    :param crop_size: size of each crop
    :param stride: stride at which crops should be taken. Must be not larger than crop size.
    :return: list of FaceCandidate objects
    """

    if crop_size < stride:

        raise ValueError("Crop size ({}) must be not smaller than stride size ({})".format(crop_size, stride))

    face_candidates = []

    offset = (crop_size - stride) // 2

    y = 0

    while y + crop_size <= image.shape[0]:

        x = 0

        while x + crop_size <= image.shape[1]:

            crop_coordinates = shapely.geometry.box(x, y, x + crop_size, y + crop_size)
            cropped_image = image[y:y + crop_size, x:x + crop_size]

            focus_coordinates = shapely.geometry.box(x + offset, y + offset, x + crop_size - offset, y + crop_size - offset)

            candidate = FaceCandidate(crop_coordinates, cropped_image, focus_coordinates)
            face_candidates.append(candidate)

            x += stride

        y += stride

    return face_candidates


def get_face_candidates_generator(image, crop_size, stride, batch_size):
    """
    Returns a generator that outputs batches of crop_size x crop_size images, each crop taken a stride away from
    previous crop.
    :param image: image from which crops are to be taken
    :param crop_size: size of each crop
    :param stride: stride at which crops should be taken. Must be not larger than crop size.
    :param batch_size: size of each batch returned by generator
    :return: generator
    """

    if crop_size < stride:

        raise ValueError("Crop size ({}) must be not smaller than stride size ({})".format(crop_size, stride))

    face_candidates = []

    offset = (crop_size - stride) // 2

    y = 0

    while y + crop_size <= image.shape[0]:

        x = 0

        while x + crop_size <= image.shape[1]:

            crop_coordinates = shapely.geometry.box(x, y, x + crop_size, y + crop_size)
            cropped_image = image[y:y + crop_size, x:x + crop_size]

            focus_coordinates = shapely.geometry.box(x + offset, y + offset, x + crop_size - offset, y + crop_size - offset)

            candidate = FaceCandidate(crop_coordinates, cropped_image, focus_coordinates)
            face_candidates.append(candidate)

            if len(face_candidates) == batch_size:

                yield(face_candidates)
                face_candidates = []

            x += stride

        y += stride

    # Yield final batch if it is non-empty, else return
    if len(face_candidates) > 0:

        yield face_candidates

    else:

        return


class SingleScaleHeatmapComputer:
    """
    Class for computing face presence heatmap given an image, prediction model and scanning parameters.
    Heatmap is computing only at a single scale.
    """

    def __init__(self, image, model, configuration):
        """
        Constructor
        :param image: image to compute heatmap for
        :param model: face prediction model
        :param configuration: FaceSearchConfiguration instance
        """

        self.image = image
        self.model = model
        self.configuration = configuration

    def get_heatmap(self):
        """
        Returns heatmap
        :return: 2D numpy array of same size as image used to construct class HeatmapComputer instance
        """

        heatmap = np.zeros(shape=self.image.shape[:2], dtype=np.float32)

        face_candidates_generator = get_face_candidates_generator(
            self.image, self.configuration.crop_size, self.configuration.stride, self.configuration.batch_size)

        for candidates_batch in face_candidates_generator:

            scores = self._get_candidate_scores(candidates_batch)

            for face_candidate, score in zip(candidates_batch, scores):

                x_start, y_start, x_end, y_end = [int(value) for value in face_candidate.focus_coordinates.bounds]
                heatmap[y_start:y_end, x_start:x_end] = score

        return heatmap

    def _get_candidate_scores(self, face_candidates):

        face_crops = [candidate.cropped_image for candidate in face_candidates]
        scores = self.model.predict(np.array(face_crops), batch_size=self.configuration.batch_size)

        return scores


class HeatmapComputer:
    """
    Class for computing face presence heatmap given an image, prediction model and scanning parameters.
    Heatmap is computed at multiple scales as per configuration parameter.
    """

    def __init__(self, image, model, configuration):
        """
        Constructor
        :param image: image to compute heatmap for
        :param model: face prediction model
        :param configuration: MultiScaleFaceSearchConfiguration instance
        """

        self.image = image
        self.model = model
        self.configuration = configuration

    def get_heatmap(self):
        """
        Returns heatmap
        :return: 2D numpy array of same size as image used to construct class HeatmapComputer instance
        """

        heatmap = np.zeros(shape=self.image.shape[:2], dtype=np.float32)

        image = self._get_largest_scale_image()

        while min(image.shape[:2]) > self.configuration.crop_size:

            image = face.processing.get_scaled_image(image, self.configuration.image_rescaling_ratio)

            single_scale_heatmap = SingleScaleHeatmapComputer(image, self.model, self.configuration).get_heatmap()
            rescaled_single_scale_heatmap = cv2.resize(single_scale_heatmap, (heatmap.shape[1], heatmap.shape[0]))

            heatmap = np.maximum(heatmap, rescaled_single_scale_heatmap)

        return heatmap

    def _get_largest_scale_image(self):

        # Get smallest size at which we want to search for a face in the image
        smallest_face_size = face.processing.get_smallest_expected_face_size(
            image_shape=self.image.shape, min_face_size=self.configuration.min_face_size,
            min_face_to_image_ratio=self.configuration.min_face_to_image_ratio)

        scale = self.configuration.crop_size / smallest_face_size

        return face.processing.get_scaled_image(self.image, scale)


def get_unique_face_detections(face_detections):
    """
    Given a list of FaceDetection objects, return only unique face detections, filtering out similar detections
    so that for each group of similar detections only a single one remains in output list
    :param face_detections: list of FaceDetection objects
    :return: list of FaceDetection objects
    """

    unique_detections = []

    for detection in face_detections:

        unique_id = 0
        similar_detection_found = False

        while unique_id < len(unique_detections) and similar_detection_found is False:

            unique_detection = unique_detections[unique_id]

            if face.geometry.get_intersection_over_union(detection.bounding_box, unique_detection.bounding_box) > 0.3:

                unique_detections[unique_id] = unique_detection \
                    if unique_detection.score > detection.score else detection

                similar_detection_found = True

            unique_id += 1

        if similar_detection_found is False:

            unique_detections.append(detection)

    return unique_detections


class FaceDetector:
    """
    Class for detecting faces in images. Given an image, prediction model and scanning parameters,
    returns a list of FaceDetection instances.
    """

    def __init__(self, image, model, configuration):
        """
        Constructor
        :param image: image to search
        :param model: face detection model
        :param configuration: FaceSearchConfiguration instance
        """

        self.image = image
        self.model = model
        self.configuration = configuration

    def get_face_detections(self):
        """
        Get face detections found in image instance was constructed with. Search is performed at a single scale.
        :return: a list of FaceDetection instances
        """

        face_detections = []

        face_candidates_generator = get_face_candidates_generator(
            self.image, self.configuration.crop_size, self.configuration.stride, self.configuration.batch_size)

        for candidates_batch in face_candidates_generator:

            scores = self._get_candidate_scores(candidates_batch)
            face_detections.extend(self._get_positive_detections(candidates_batch, scores))

        return get_unique_face_detections(face_detections)

    def _get_candidate_scores(self, face_candidates):

        face_crops = [candidate.cropped_image for candidate in face_candidates]
        scores = self.model.predict(np.array(face_crops), batch_size=self.configuration.batch_size)

        return scores

    def _get_positive_detections(self, face_candidates, scores):

        face_detections = []

        for candidate, score in zip(face_candidates, scores):

            if score > 0.5:

                detection = FaceDetection(candidate.crop_coordinates, score)
                face_detections.append(detection)

        return face_detections


class MultiScaleFaceDetector:
    """
    Class for detecting faces in images. Faces are search for at multiple scales,
     as per configuration parameters.
    """

    def __init__(self, image, model, configuration):
        """
        Constructor
        :param image: image to search
        :param model: face detection model
        :param configuration: MultiScaleFaceSearchConfiguration instance
        """

        self.image = image
        self.model = model
        self.configuration = configuration

    def get_faces_detections(self):
        """
        Get face detections found in image instance was constructed with. Search is performed at a single scale.
        :return: a list of FaceDetection instances
        """

        image = self._get_largest_scale_image()

        detections = []

        while min(image.shape[:2]) > self.configuration.crop_size:

            current_scale_detections = FaceDetector(image, self.model, self.configuration).get_face_detections()
            detections = get_unique_face_detections(detections + current_scale_detections)

            image = face.processing.get_scaled_image(image, self.configuration.image_rescaling_ratio)

        return get_unique_face_detections(detections)

    def _get_largest_scale_image(self):

        # Get smallest size at which we want to search for a face in the image
        smallest_face_size = face.processing.get_smallest_expected_face_size(
            image_shape=self.image.shape, min_face_size=self.configuration.min_face_size,
            min_face_to_image_ratio=self.configuration.min_face_to_image_ratio)

        scale = self.configuration.crop_size / smallest_face_size

        return face.processing.get_scaled_image(self.image, scale)
