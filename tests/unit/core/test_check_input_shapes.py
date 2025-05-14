"""Tests the check_input_shapes function."""

from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest


@pytest.mark.parametrize(
    argnames="inputs, raises",
    argvalues=[
        ([0], does_not_raise()),
        ([None], does_not_raise()),
        ([np.array([])], does_not_raise()),
        ([np.ones(3)], does_not_raise()),
        ([np.ones((3, 2)), np.ones((3, 2))], does_not_raise()),
        ([np.ones((1, 2)), np.ones((3, 3))], pytest.raises(ValueError)),
        ([np.ones((1, 2)), np.ones((3, 1))], does_not_raise()),
        ([np.ones(2), np.ones((3, 2))], does_not_raise()),
        ([np.ones(3), np.ones((3, 2))], pytest.raises(ValueError)),
    ],
)
def test_check_input_shapes(inputs, raises):
    """Tests if the inputs satisfy check_input_shapes."""

    from pyrealm.core.utilities import check_input_shapes

    with raises:
        check_input_shapes(*inputs)
