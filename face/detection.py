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

    def get_scaled(self, scale):

        rescaled_bounding_box = face.geometry.get_scaled_bounding_box(self.bounding_box, scale)
        return FaceDetection(rescaled_bounding_box, self.score)


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

                x_start, y_start, x_end, y_end = [round(value) for value in face_candidate.focus_coordinates.bounds]
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


class UniqueDetectionsComputer:
    """
    A simple static class with different functions for computing unique detections
    """

    @staticmethod
    def non_maximum_suppression(face_detections, iou_threshold):
        """
        Given a list of FaceDetection objects, return only detections that
        represent a local maximum.
        :param face_detections: list of FaceDetection objects
        :param iou_threshold: value above which IOU of two detections must be for them to be considered similar
        :return: list of FaceDetection objects
        """

        unique_detections = []

        for detection in face_detections:

            unique_id = 0
            similar_detection_found = False

            while unique_id < len(unique_detections) and similar_detection_found is False:

                unique_detection = unique_detections[unique_id]

                if face.geometry.get_intersection_over_union(
                        detection.bounding_box, unique_detection.bounding_box) > iou_threshold:

                    unique_detections[unique_id] = unique_detection \
                        if unique_detection.score > detection.score else detection

                    similar_detection_found = True

                unique_id += 1

            if similar_detection_found is False:

                unique_detections.append(detection)

        return unique_detections

    @staticmethod
    def averaging(face_detections, iou_threshold):
        """
        Given a list of FaceDetection objects, group detections that have high IOU threshold together, compute
        their average and return averages of each group as unique detections. Averages have scores of the
        highest scored detection of the group they come from.
        :param face_detections: list of FaceDetection objects
        :param iou_threshold: value above which IOU of two detections must be for them to be considered similar
        :return: list of FaceDetection objects
        """

        groups_list = []

        # First get groups of similar detections
        for detection in face_detections:

            group_found = False

            group_id = 0

            while group_id < len(groups_list) and group_found is False:

                current_group = groups_list[group_id]
                member_id = 0

                while member_id < len(current_group) and group_found is False:

                    group_member = current_group[member_id]

                    if face.geometry.get_intersection_over_union(
                            detection.bounding_box, group_member.bounding_box) > iou_threshold:

                        current_group.append(detection)
                        group_found = True

                    member_id += 1

                group_id += 1

            if group_found is False:

                group = [detection]
                groups_list.append(group)

        unique_detections = []

        # Then get average detection for each group
        for group in groups_list:

            coordinates = np.array([detection.bounding_box.bounds for detection in group])
            average_coordinates = np.mean(coordinates, axis=0)

            int_coordinates = [round(coordinate) for coordinate in average_coordinates]

            score = max([detection.score for detection in group])

            average_detection = FaceDetection(shapely.geometry.box(*int_coordinates), score)
            unique_detections.append(average_detection)

        return unique_detections


class SingleScaleFaceDetector:
    """
    Class for detecting faces in images at a single scale. Given an image, prediction model and scanning parameters,
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

        return UniqueDetectionsComputer.averaging(face_detections, iou_threshold=0.2)

    def _get_candidate_scores(self, face_candidates):

        face_crops = [candidate.cropped_image for candidate in face_candidates]
        scores = self.model.predict(np.array(face_crops), batch_size=self.configuration.batch_size)

        return scores

    def _get_positive_detections(self, face_candidates, scores):

        face_detections = []

        for candidate, score in zip(face_candidates, scores):

            if score > 0.9:

                detection = FaceDetection(candidate.crop_coordinates, score)
                face_detections.append(detection)

        return face_detections


class FaceDetector:
    """
    Class for detecting faces in images. Faces are searched for at multiple scales,
     as per configuration parameters.
    """

    def __init__(self, image, model, configuration):
        """
        Constructor
        :param image: image to search
        :param model: face detection model
        :param configuration: MultiScaleFaceSearchConfiguration instance
        """

        # Scale image down if it is too large
        self.input_image_scale = 1 if min(image.shape[:2]) < 500 else 500 / min(image.shape[:2])
        self.image = face.processing.get_scaled_image(image, self.input_image_scale)

        self.model = model
        self.configuration = configuration

    def get_faces_detections(self):
        """
        Get face detections found in image instance was constructed with. Search is performed at a single scale.
        :return: a list of FaceDetection instances
        """

        current_scale = self._get_largest_scale()
        image = face.processing.get_scaled_image(self.image, current_scale)

        detections = []

        while min(image.shape[:2]) > self.configuration.crop_size:

            current_scale_detections = SingleScaleFaceDetector(
                image, self.model, self.configuration).get_face_detections()

            rescaled_detections = [detection.get_scaled(1 / current_scale) for detection in current_scale_detections]
            detections.extend(rescaled_detections)

            current_scale *= self.configuration.image_rescaling_ratio
            image = face.processing.get_scaled_image(self.image, current_scale)

        # Get unique detections and scale them as necessary, since input image might have been scaled
        unique_detections = UniqueDetectionsComputer.averaging(detections, iou_threshold=0.2)
        return [detection.get_scaled(1 / self.input_image_scale) for detection in unique_detections]

    def _get_largest_scale(self):

        # Get smallest size at which we want to search for a face in the image
        smallest_face_size = face.processing.get_smallest_expected_face_size(
            image_shape=self.image.shape, min_face_size=self.configuration.min_face_size,
            min_face_to_image_ratio=self.configuration.min_face_to_image_ratio)

        return self.configuration.crop_size / smallest_face_size
