"""
Module with various utilities, mostly files related
"""


def get_file_lines(path):
    """
    Get file lines. This function wraps opening file, reading lines and closing file in one operation.
    :param path: path to file
    :return: list of lines
    """

    with open(path) as file:
        return file.readlines()


def get_file_line_count(path):
    """
    Give a path to a file, return number of lines file has
    :param path: path to file
    :return: number of lines file has
    """

    return len(get_file_lines(path))