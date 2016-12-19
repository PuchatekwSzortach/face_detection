"""
Module with utilities for downloading data
"""


class Downloader:
    """
    A simple class supports downloading large files with retries.
    """

    def __init__(self, url, path, max_retries=5):
        """
        Constructor
        :param url: url to download from
        :param path: path to save downloaded file to
        :param max_retries: max number of retries should download fail
        """

        self.url = url
        self.path = path

        self.max_retries = max_retries
        self.reties_count = 0

    def download(self):

        print("{} to {}".format(self.url, self.path))
