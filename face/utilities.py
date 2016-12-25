"""
Module with various utilities, mostly files related
"""

import logging
import os

import cv2


def get_file_lines(path):
    """
    Get file lines. This function wraps opening file, reading lines and closing file in one operation.
    :param path: path to file
    :return: list of lines
    """

    with open(path) as file:
        return file.readlines()


def get_file_lines_count(path):
    """
    Give a path to a file, return number of lines file has
    :param path: path to file
    :return: number of lines file has
    """

    return len(get_file_lines(path))


def get_logger():
    """
    Returns a logger that writes to an html file in /tmp directory
    :return: logger instance
    """

    directory = "/tmp/faces"
    path = os.path.join(directory, "log.html")

    os.makedirs(directory, exist_ok=True)

    logger = logging.getLogger("faces")
    file_handler = logging.FileHandler(path, mode="w")

    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger


def get_image(path):
    """
    Get image at a given path, applying any necessary scaling.
    :param path:
    :return: numpy array
    """

    return cv2.imread(path) / 255
