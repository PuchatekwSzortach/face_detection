"""
Tests for face.download module
"""

import mock

import face.download


def test_get_url_asset_size():

    mock_urlopener = mock.mock_open()

    context = mock_urlopener.return_value.__enter__.return_value
    context.info = mock.Mock(return_value={"Content-Length": "10"})

    size = face.download.get_url_asset_size(url="url", urlopen=mock_urlopener)
    assert 10 == size
