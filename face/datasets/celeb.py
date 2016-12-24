"""
Code for working with Celeb+ dataset
"""

import os
import shutil
import subprocess
import glob

import shapely.geometry

import face.download
import face.utilities


class DatasetBuilder:
    """
    Class for downloading Celeb+ data and preparing datasets from it.
    """

    def __init__(self, data_directory):

        self.data_directory = data_directory
        self.bounding_boxes_path = os.path.join(self.data_directory, "all_bounding_boxes.txt")

    def build_datasets(self):

        # shutil.rmtree(self.data_directory, ignore_errors=True)
        # os.makedirs(self.data_directory, exist_ok=True)

        # self._get_images()
        # self._get_bounding_boxes()

        # image_paths = self._get_image_paths(self.data_directory)
        # bounding_boxes_map = self._get_bounding_boxes_map(self.bounding_boxes_path)

    def _get_images(self):

        image_archives_urls = [
            "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AABQwEE5YX5jTFGXjo0f9glIa/Img/img_celeba.7z/img_celeba.7z.001?dl=1",
            "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADxKopMA7g_Ka2o7X7B8jiHa/Img/img_celeba.7z/img_celeba.7z.002?dl=1",
            "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AABSqeGALxGo1sXZ-ZizRFa5a/Img/img_celeba.7z/img_celeba.7z.003?dl=1",
            "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADBal8W3N9AYwYuqwTtA_fQa/Img/img_celeba.7z/img_celeba.7z.004?dl=1",
            "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AACJaDb7rWNFcCKqcFjFjUlHa/Img/img_celeba.7z/img_celeba.7z.005?dl=1",
            "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AACcD0ZMO36zVaIfLGLKtrq4a/Img/img_celeba.7z/img_celeba.7z.006?dl=1",
            "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAAhuX-S5ULmy8GII6jlZFb9a/Img/img_celeba.7z/img_celeba.7z.007?dl=1",
            "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAAUtign0NJIV8fRK7xt6TIEa/Img/img_celeba.7z/img_celeba.7z.008?dl=1",
            "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AACJsmneLOU5xMB2qmnJA0AGa/Img/img_celeba.7z/img_celeba.7z.009?dl=1",
            "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAAfZVSjBlkPr5e5GYMek50_a/Img/img_celeba.7z/img_celeba.7z.010?dl=1",
            "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAA6-edxuJyMBoGZqTdl28bpa/Img/img_celeba.7z/img_celeba.7z.011?dl=1",
            "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AABMLOgnvv8DKxt4UvULSAoha/Img/img_celeba.7z/img_celeba.7z.012?dl=1",
            "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AABOeeqqAzZEY6jDwTdOUTqRa/Img/img_celeba.7z/img_celeba.7z.013?dl=1",
            "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADuEM2h2qG_L0UbUTViRH5Da/Img/img_celeba.7z/img_celeba.7z.014?dl=1"
        ]

        filenames = [os.path.basename(url).split("?")[0] for url in image_archives_urls]
        paths = [os.path.join(self.data_directory, filename) for filename in filenames]

        # Download image archives
        for url, path in zip(image_archives_urls, paths):

            face.download.Downloader(url, path).download()

        # Extract images
        subprocess.call(["7z", "x", paths[0], "-o" + self.data_directory])

        # Delete image archives
        for path in paths:

            os.remove(path)

    def _get_bounding_boxes(self):

        url = "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AACL5lLyHMAHvFA8W17JDahma/Anno/list_bbox_celeba.txt?dl=1"
        face.download.Downloader(url, self.bounding_boxes_path).download()

    def _get_image_paths(self, data_directory):

        image_paths = glob.glob(os.path.join(data_directory, "**/*.jpg"), recursive=True)
        image_paths = [os.path.abspath(path) for path in image_paths]
        return image_paths

    def _get_bounding_boxes_map(self, bounding_boxes_path):

        bounding_boxes_lines = face.utilities.get_file_lines(bounding_boxes_path)[2:]

        bounding_boxes_map = {}

        for line in bounding_boxes_lines:

            tokens = line.split()

            filename = tokens[0]
            bounding_box = self._get_bounding_box_from_tokens(tokens[1:])

            bounding_boxes_map[filename] = bounding_box

        return bounding_boxes_map

    def _get_bounding_box_from_tokens(self, tokens):

        upper_left_point = (tokens[0], tokens[1])
        upper_right_point = (tokens[0] + tokens[2], tokens[1])
        lower_right_point = (tokens[0] + tokens[2], tokens[1] + tokens[3])
        lower_left_point = (tokens[0], tokens[1] + tokens[3])

        bounding_box = shapely.geometry.Polygon(
            [upper_left_point, upper_right_point, lower_right_point, lower_left_point])

        return bounding_box









