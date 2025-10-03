"""Tests the broadcast_time function from time_series.py."""

from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest


@pytest.mark.parametrize(
    argnames="values,shape,expected,raises",
    argvalues=(
        (np.ones(1), (3,), np.ones(3), does_not_raise()),
        (np.ones(3), (3,), np.ones(3), does_not_raise()),
        (np.ones((1, 2)), (3, 2), np.ones((3, 2)), does_not_raise()),
        (np.ones((1, 2)), (3, 4), np.ones((3, 2)), does_not_raise()),
        (np.ones(2), (3, 2, 2), np.ones((3, 1, 2)), does_not_raise()),
        (np.ones(4), (3,), None, pytest.raises(ValueError)),
        (np.ones((2, 2)), (2,), None, pytest.raises(ValueError)),
    ),
)
def test_broadcast_time(values, shape, expected, raises):
    """Test correct outputs and errors for the broadcast_time function."""
    from pyrealm.core.time_series import broadcast_time

    with raises:
        result = broadcast_time(values, shape)
        assert np.array_equal(result, expected)
