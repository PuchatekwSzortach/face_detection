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

    def __init__(self, url, path, max_retries=5, **kwargs):
        """
        Constructor
        :param url: url to download from
        :param path: path to save downloaded file to
        :param max_retries: max number of retries should download fail
        """

        self.url = url
        self.path = path

        self.max_retries = max_retries

        self.url_opener = kwargs["url_opener"] if "url_opener" in kwargs else urllib.request.urlopen
        self.url_request = kwargs["url_request"] if "url_request" in kwargs else urllib.request.Request
        self.file_opener = kwargs["file_opener"] if "file_opener" in kwargs else open

        self.reties_count = 0
        self.downloaded_bytes_count = 0

        self.total_bytes_count = get_url_asset_size(self.url, url_opener=self.url_opener)

        self.bytes_per_read = 8192

    def download(self, verbose=True):
        """
        Start download
        :param verbose: whether to output progress messages
        """

        if verbose:
            print("Downloading {}".format(self.url))

        try:

            request = self._get_request()
            flags = "wb" if self.downloaded_bytes_count is 0 else "ab"

            with self.url_opener(request) as url_connection, self.file_opener(self.path, mode=flags) as file, \
                    tqdm.tqdm(total=self.total_bytes_count, disable=not verbose) as progress_bar:

                progress_bar.update(self.downloaded_bytes_count)
                data = url_connection.read(self.bytes_per_read)

                # Read while data is available
                while len(data) != 0:

                    file.write(data)
                    self.downloaded_bytes_count += len(data)

                    progress_bar.update(len(data))
                    data = url_connection.read(self.bytes_per_read)

                # Sometimes server sends empty packet even though not all data has been downloaded yet
                if self.downloaded_bytes_count != self.total_bytes_count:

                    raise urllib.error.ContentTooShortError(
                        message="Empty packet received before end of download",
                        content=data)

        except (TimeoutError, urllib.error.ContentTooShortError) as error:

            self._handle_error(error, verbose)

    def _get_request(self):

        header = {'Range': 'bytes={}-{}'.format(self.downloaded_bytes_count, self.total_bytes_count)}
        return self.url_request(url=self.url, headers=header)

    def _handle_error(self, error, verbose):

        if self.reties_count < self.max_retries:

            if verbose:
                print("Download failed, retrying...")

            self.reties_count += 1
            self.download(verbose)

        else:

            if verbose:
                print("Download failed despite retrying {} times, raising error".format(self.reties_count))

            raise error
