"""
Tests for face.utilities module
"""

import face.utilities


def test_get_batches_even_split():

    data = list(range(9))
    batch_size = 3

    expected = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    actual = face.utilities.get_batches(data, batch_size)

    assert expected == actual


def test_get_batches_uneven_split():

    data = list(range(10))
    batch_size = 3

    expected = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    actual = face.utilities.get_batches(data, batch_size)

    assert expected == actual