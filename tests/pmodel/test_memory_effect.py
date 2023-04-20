import numpy as np
import pytest


@pytest.mark.parametrize(
    argnames="inputs",
    argvalues=[
        pytest.param(np.arange(0, 10), id="1D"),
        pytest.param(
            np.column_stack([np.arange(0, 10)] * 4) + np.arange(4),
            id="2D",
        ),
        pytest.param(
            np.dstack([np.column_stack([np.arange(0, 10)] * 4)] * 4)
            + np.arange(16).reshape(4, 4),
            id="3D",
        ),
    ],
)
@pytest.mark.parametrize(argnames="alpha", argvalues=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
def test_memory_effect(inputs, alpha):
    """Test the memory effect.

    This uses matrix maths to calculate expected by calculating the coefficients of the
    recursive equation. Does not scale well, but a useful parallel implementation for
    testing.
    """
    from pyrealm.pmodel import memory_effect

    result = memory_effect(inputs, alpha=alpha)

    # Calculate the coefficients for product sum of the elements along the time axis
    one_minus_alpha = 1 - alpha
    n = len(inputs)
    ident = np.identity(n)
    nan = np.ones_like(ident) * np.nan
    rw, cl = np.indices(ident.shape)
    rwcl = rw - cl
    one_minus_alpha_exp = np.where(rwcl >= 0, rwcl, nan)
    alpha_exp = np.where(rwcl >= 0, 1, nan)
    alpha_exp[:, 0] = 0

    coef = one_minus_alpha**one_minus_alpha_exp * alpha**alpha_exp
    coef = np.where(np.triu(coef, k=1), 0, coef)

    # Calculate the tensor dot product of the coefficients and the inputs along the
    # first (time) axis.
    expected = np.tensordot(coef, inputs, axes=1)

    assert np.allclose(result, expected)
