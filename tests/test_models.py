"""Tests for statistics functions within the Model layer."""
import pytest

import numpy as np
import numpy.testing as npt


def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""
    from inflammation.models import daily_mean

    test_input = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[1, 2], [3, 4], [5, 6]], [3, 4]),
        ([[1, 9], [1, 4], [10, 6]], [4.0, 6.333333]),
    ])
def test_daily_mean_integers(test, expected):
    """Test that mean function works for an array of positive integers."""
    from inflammation.models import daily_mean

    # Need to use Numpy testing functions to compare arrays
    assert np.allclose(daily_mean(np.array(test)), np.array(expected))

