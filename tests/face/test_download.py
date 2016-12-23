"""
Tests for face.download module
"""
import itertools

import mock
import urllib.error

import pytest

import face.download


def test_get_url_asset_size():

    mock_url_opener = mock.mock_open()

    context = mock_url_opener.return_value.__enter__.return_value
    context.info = mock.Mock(return_value={"Content-Length": "10"})

    size = face.download.get_url_asset_size(url="url", url_opener=mock_url_opener)
    assert 10 == size


class TestDownloader:
    """
    Test class for face.download.Downloader
    """

    def setup_method(self, method):

        self.mock_url_opener = mock.mock_open()
        self.mock_url_context = self.mock_url_opener.return_value.__enter__.return_value

        self.mock_file_opener = mock.mock_open()
        self.mock_file_context = self.mock_file_opener.return_value.__enter__.return_value

        self.mock_url_request = mock.Mock()

    def test_downloader_simple_one_read_download(self):

        self.mock_url_context.info = mock.Mock(return_value={"Content-Length": "10"})

        # Data return by context
        packet = 10 * [1]
        self.mock_url_context.read.side_effect = [packet, []]

        downloader = face.download.Downloader(
            url="whatever", path="whatever", url_opener=self.mock_url_opener)

        downloader.download(
            url_request=self.mock_url_request, url_opener=self.mock_url_opener,
            file_opener=self.mock_file_opener)

        assert 2 == self.mock_url_opener.call_count
        assert 1 == self.mock_url_request.call_count
        assert 1 == self.mock_file_opener.call_count

        self.mock_file_context.write.assert_called_once_with(packet)

        assert 10 == downloader.downloaded_bytes_count

    def test_downloader_simple_download_over_three_calls(self):

        self.mock_url_context.info = mock.Mock(return_value={"Content-Length": "30"})

        # Data return by context
        packet = 10 * [1]
        self.mock_url_context.read.side_effect = [packet, packet, packet, []]

        downloader = face.download.Downloader(
            url="whatever", path="whatever", url_opener=self.mock_url_opener)

        downloader.download(
            url_request=self.mock_url_request, url_opener=self.mock_url_opener,
            file_opener=self.mock_file_opener)

        assert 2 == self.mock_url_opener.call_count
        assert 1 == self.mock_url_request.call_count
        assert 1 == self.mock_file_opener.call_count

        calls = itertools.repeat(mock.call(packet), 3)
        self.mock_file_context.write.assert_has_calls(calls)
        assert 3 == self.mock_file_context.write.call_count

        assert 30 == downloader.downloaded_bytes_count

    def test_downloader_with_timeout_error(self):

        self.mock_url_context.info = mock.Mock(return_value={"Content-Length": "30"})

        # Data return by context
        packet = 10 * [1]
        self.mock_url_context.read.side_effect = [packet, TimeoutError(), packet, packet, []]

        downloader = face.download.Downloader(
            url="whatever", path="whatever", url_opener=self.mock_url_opener)

        downloader.download(
            url_request=self.mock_url_request, url_opener=self.mock_url_opener,
            file_opener=self.mock_file_opener)

        assert 3 == self.mock_url_opener.call_count
        assert 2 == self.mock_url_request.call_count
        assert 2 == self.mock_file_opener.call_count

        calls = itertools.repeat(mock.call(packet), 3)
        self.mock_file_context.write.assert_has_calls(calls)
        assert 3 == self.mock_file_context.write.call_count

        assert 30 == downloader.downloaded_bytes_count

    def test_downloader_with_timeout_error_over_max_tries(self):

        self.mock_url_context.info = mock.Mock(return_value={"Content-Length": "30"})

        # Data return by context
        packet = 10 * [1]
        self.mock_url_context.read.side_effect = [
            packet, TimeoutError(), TimeoutError(), packet, TimeoutError(), TimeoutError(), packet, []]

        downloader = face.download.Downloader(max_retries=3, url="whatever", path="whatever",
                                              url_opener=self.mock_url_opener)

        with pytest.raises(TimeoutError):
            downloader.download(
                url_request=self.mock_url_request, url_opener=self.mock_url_opener,
                file_opener=self.mock_file_opener)

        assert 5 == self.mock_url_opener.call_count
        assert 4 == self.mock_url_request.call_count
        assert 4 == self.mock_file_opener.call_count

        calls = itertools.repeat(mock.call(packet), 2)
        self.mock_file_context.write.assert_has_calls(calls)
        assert 2 == self.mock_file_context.write.call_count

        assert 20 == downloader.downloaded_bytes_count
