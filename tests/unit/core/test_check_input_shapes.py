"""Tests the check_input_shapes function."""

from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
import xarray as xr


@pytest.mark.parametrize("array_type", [np.array, xr.DataArray])
@pytest.mark.parametrize(
    argnames="inputs, shape, raises",
    argvalues=[
        ([0], None, does_not_raise()),
        ([None], None, does_not_raise()),
        ([np.array([])], None, does_not_raise()),
        ([np.ones(3)], None, does_not_raise()),
        ([np.ones((3, 2)), np.ones((3, 2))], None, does_not_raise()),
        ([np.ones((1, 2)), np.ones((3, 2))], None, does_not_raise()),
        ([np.ones((1, 2)), np.ones((3, 1))], None, does_not_raise()),
        ([np.ones((1, 2)), np.ones((3, 3))], None, pytest.raises(ValueError)),
        ([np.ones((3, 1, 4)), np.ones((3, 2, 4))], None, does_not_raise()),
        ([np.ones((1, 1, 4)), np.ones((3, 2, 4))], None, does_not_raise()),
        ([np.ones(2), np.ones((3, 2))], None, pytest.raises(ValueError)),
        ([np.ones(3), np.ones((3, 2))], None, pytest.raises(ValueError)),
        ([0], (1,), does_not_raise()),
        ([0], (3, 2), does_not_raise()),
        (np.array([]), (1,), does_not_raise()),
        ([np.ones((1, 2)), np.ones((3, 1))], (3, 2), does_not_raise()),
        ([np.ones((1, 2)), np.ones((3, 1))], (1, 3, 2), pytest.raises(ValueError)),
    ],
)
def test_check_input_shapes(array_type, inputs, shape, raises):
    """Tests if the inputs satisfy check_input_shapes."""

    from pyrealm.core.utilities import check_input_shapes

    for i, val in enumerate(inputs):
        if isinstance(val, np.ndarray):
            inputs[i] = array_type(val)

    with raises:
        check_input_shapes(*inputs, shape=shape)
