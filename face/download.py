"""
Module with utilities for downloading data
"""

import urllib.request
import urllib.error

import tqdm


def get_url_asset_size(url, url_opener=urllib.request.urlopen):
    """
    Get size of asset under a url
    :param url: url to look for
    :param urlopen: function used to open url, defaults to urllib.request.urlopen
    :return: asset size in bytes
    """

    with url_opener(url) as url_connection:

        metadata = url_connection.info()
        return int(metadata["Content-Length"])


class Downloader:
    """
    A simple class supports downloading large files with retries.
    """

    def __init__(self, url, path, max_retries=5, url_opener=urllib.request.urlopen):
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
        self.total_bytes_count = get_url_asset_size(self.url, url_opener=url_opener)

        self.bytes_per_read = 8192

    def download(self, url_request=urllib.request.Request, url_opener=urllib.request.urlopen, file_opener=open):

        print("Downloading {}".format(self.url))

        header = {'Range': 'bytes={}-{}'.format(self.downloaded_bytes_count, self.total_bytes_count)}
        request = url_request(url=self.url, headers=header)

        flags = "wb" if self.downloaded_bytes_count is 0 else "ab"

        try:

            with url_opener(request) as url_connection, file_opener(self.path, mode=flags) as file, \
                    tqdm.tqdm(total=self.total_bytes_count) as progress_bar:

                progress_bar.update(self.downloaded_bytes_count)

                data = url_connection.read(self.bytes_per_read)

                print()
                print("Data is: {} and has length {}".format(data, len(data)))

                while len(data) != 0:

                    file.write(data)
                    self.downloaded_bytes_count += len(data)
                    progress_bar.update(len(data))

                    data = url_connection.read(self.bytes_per_read)

        except TimeoutError as error:

            if self.reties_count < self.max_retries:

                self.reties_count += 1
                self.download(url_request, url_opener, file_opener)

            else:

                raise error
