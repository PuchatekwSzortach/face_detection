"""
Tests for face.download module
"""

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


def test_downloader_simple_one_read_download():

    mock_url_opener = mock.mock_open()
    mock_file_opener = mock.mock_open()

    # Mock url opener context
    context = mock_url_opener.return_value.__enter__.return_value
    context.info = mock.Mock(return_value={"Content-Length": "10"})

    # Data return by context
    packet = 10 * [1]
    context.read.side_effect = [packet, []]
    mock_url_request = mock.Mock()

    downloader = face.download.Downloader(url="whatever", path="whatever", url_opener=mock_url_opener)
    downloader.download(url_request=mock_url_request, url_opener=mock_url_opener, file_opener=mock_file_opener)

    assert 2 == mock_url_opener.call_count
    assert 1 == mock_url_request.call_count
    assert 1 == mock_file_opener.call_count

    assert 10 == downloader.downloaded_bytes_count


def test_downloader_simple_download_over_three_calls():

    mock_url_opener = mock.mock_open()
    mock_file_opener = mock.mock_open()

    # Mock url opener context
    context = mock_url_opener.return_value.__enter__.return_value
    context.info = mock.Mock(return_value={"Content-Length": "30"})

    # Data return by context
    packet = 10 * [1]
    context.read.side_effect = [packet, packet, packet, []]

    mock_url_request = mock.Mock()

    downloader = face.download.Downloader(url="whatever", path="whatever", url_opener=mock_url_opener)
    downloader.download(url_request=mock_url_request, url_opener=mock_url_opener, file_opener=mock_file_opener)

    assert 2 == mock_url_opener.call_count
    assert 1 == mock_url_request.call_count
    assert 1 == mock_file_opener.call_count

    assert 30 == downloader.downloaded_bytes_count


def test_downloader_with_timeout_error():

    mock_url_opener = mock.mock_open()
    mock_file_opener = mock.mock_open()

    # Mock url opener context
    context = mock_url_opener.return_value.__enter__.return_value
    context.info = mock.Mock(return_value={"Content-Length": "30"})

    # Data return by context
    packet = 10 * [1]
    context.read.side_effect = [packet, TimeoutError(), packet, packet, []]

    mock_url_request = mock.Mock()

    downloader = face.download.Downloader(url="whatever", path="whatever", url_opener=mock_url_opener)
    downloader.download(url_request=mock_url_request, url_opener=mock_url_opener, file_opener=mock_file_opener)

    assert 3 == mock_url_opener.call_count
    assert 2 == mock_url_request.call_count
    assert 2 == mock_file_opener.call_count

    assert 30 == downloader.downloaded_bytes_count


def test_downloader_with_timeout_error_over_max_tries():
    
    mock_url_opener = mock.mock_open()
    mock_file_opener = mock.mock_open()

    # Mock url opener context
    context = mock_url_opener.return_value.__enter__.return_value
    context.info = mock.Mock(return_value={"Content-Length": "30"})

    # Data return by context
    packet = 10 * [1]
    context.read.side_effect = [
        packet, TimeoutError(), TimeoutError(), packet, TimeoutError(), TimeoutError(), packet, []]

    mock_url_request = mock.Mock()

    downloader = face.download.Downloader(max_retries=3, url="whatever", path="whatever", url_opener=mock_url_opener)

    with pytest.raises(TimeoutError):

        downloader.download(url_request=mock_url_request, url_opener=mock_url_opener, file_opener=mock_file_opener)

    assert 5 == mock_url_opener.call_count
    assert 4 == mock_url_request.call_count
    assert 4 == mock_file_opener.call_count

    assert 20 == downloader.downloaded_bytes_count
