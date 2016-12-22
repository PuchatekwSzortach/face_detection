"""
Tests for face.download module
"""

import mock

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
    context.read.side_effect = [10 * [1], []]

    downloader = face.download.Downloader(url="whatever", path="whatever", url_opener=mock_url_opener)

    mock_url_request = mock.Mock()

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
    context.read.side_effect = [10 * [1], 10 * [1], 10 * [1], []]

    downloader = face.download.Downloader(url="whatever", path="whatever", url_opener=mock_url_opener)

    mock_url_request = mock.Mock()

    downloader.download(url_request=mock_url_request, url_opener=mock_url_opener, file_opener=mock_file_opener)

    assert 2 == mock_url_opener.call_count
    assert 1 == mock_url_request.call_count
    assert 1 == mock_file_opener.call_count

    assert 30 == downloader.downloaded_bytes_count
