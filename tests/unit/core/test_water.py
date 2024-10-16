"""Test the pmodel functions.

TODO - note that there are parallel tests in test_pmodel that benchmark against the
rpmodel outputs and test a wider range of inputs. Those could be moved here. These tests
check the size of outputs and that the results meet a simple benchmark value.
"""

import numpy as np
import pytest


@pytest.mark.parametrize(argnames="shape", argvalues=[(1,), (6, 9), (4, 7, 3)])
def test_calc_density_h20_fisher(shape):
    """Test the fisher method."""
    from pyrealm.core.water import calc_density_h2o_fisher

    rho = calc_density_h2o_fisher(
        np.full(shape, fill_value=20), np.full(shape, fill_value=101325)
    )

    assert np.allclose(rho.round(3), np.full(shape, fill_value=998.206))


@pytest.mark.parametrize(argnames="shape", argvalues=[(1,), (6, 9), (4, 7, 3)])
def test_calc_density_h20_chen(shape):
    """Test the chen method."""
    from pyrealm.core.water import calc_density_h2o_chen

    rho = calc_density_h2o_chen(
        np.full(shape, fill_value=20), np.full(shape, fill_value=101325)
    )

    assert np.allclose(rho.round(3), np.full(shape, fill_value=998.25))


@pytest.mark.parametrize(
    argnames="const_args, exp",
    argvalues=[
        pytest.param(None, 998.206, id="defaults"),
        pytest.param("fisher", 998.206, id="fisher_via_const"),
        pytest.param("chen", 998.25, id="chen_via_const"),
    ],
)
def test_calc_density_h20(const_args, exp):
    """Test the wrapper method dispatches as expected."""
    from pyrealm.constants import CoreConst
    from pyrealm.core.water import calc_density_h2o

    args = {}

    if const_args is not None:
        args["core_const"] = CoreConst(water_density_method=const_args)

    rho = calc_density_h2o(np.array([20]), np.array([101325]), **args)

    assert rho.round(3) == exp


@pytest.mark.parametrize(argnames="shape", argvalues=[(1,), (6, 9), (4, 7, 3)])
def test_calc_viscosity_h20(shape):
    """Test the viscosity calculation."""
    from pyrealm.core.water import calc_viscosity_h2o

    eta = calc_viscosity_h2o(
        np.full(shape, fill_value=20), np.full(shape, fill_value=101325)
    )

    assert np.allclose(eta.round(7), np.full(shape, fill_value=0.0010016))


@pytest.mark.parametrize(argnames="shape", argvalues=[(1,), (6, 9), (4, 7, 3)])
def test_calc_viscosity_h20_matrix(shape):
    """Test the viscosity calculation."""
    from pyrealm.core.water import calc_viscosity_h2o_matrix

    eta = calc_viscosity_h2o_matrix(
        np.full(shape, fill_value=20), np.full(shape, fill_value=101325)
    )

    assert np.allclose(eta.round(7), np.full(shape, fill_value=0.0010016))
