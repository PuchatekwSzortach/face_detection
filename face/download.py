"""
Module with utilities for downloading data
"""

import urllib.request
import urllib.error

import tqdm


def get_url_asset_size(url, urlopen=urllib.request.urlopen):
    """
    Get size of asset under a url
    :param url: url to look for
    :param urlopen: function used to open url, defaults to urllib.request.urlopen
    :return: asset size in bytes
    """

    with urlopen(url) as url_connection:

        metadata = url_connection.info()
        return int(metadata["Content-Length"])


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

        self.downloaded_bytes_count = 0
        self.total_bytes_count = get_url_asset_size(self.url)

        self.bytes_per_read = 8192

    def download(self):

        print("Downloading {}".format(self.url))

        header = {'Range': 'bytes={}-{}'.format(self.downloaded_bytes_count, self.total_bytes_count)}
        request = urllib.request.Request(url=self.url, headers=header)

        flags = "wb" if self.downloaded_bytes_count is 0 else "ab"

        with urllib.request.urlopen(request) as url_connection, open(self.path, mode=flags) as file, \
            tqdm.tqdm(total=self.total_bytes_count) as progress_bar:

            progress_bar.update(self.downloaded_bytes_count)

            bytes = url_connection.read(self.bytes_per_read)

            while len(bytes) > 0:

                file.write(bytes)
                self.downloaded_bytes_count += len(bytes)
                progress_bar.update(len(bytes))

                bytes = url_connection.read(self.bytes_per_read)
                
