import pytest
import numpy as np
import dotmap

from pypmodel import pmodel

# ------------------------------------------
# Fixtures: shared input values
# ------------------------------------------


@pytest.fixture(scope='module')
def common_inputs():
    """Parameterised fixture to run tests using both the local and remote validator.
    """

    return dotmap.DotMap(tc=20,
                         tc_mat=np.array([[15, 20], [25, 30]]),
                         tc_broadcast_err=np.array([[15, 20, 25]]),
                         p=101325,
                         p_mat=np.array([[100325, 101325], [102325, 103325]]),
                         p_broadcast_err=np.array([[100325, 101325, 103325]]))


# ------------------------------------------
# Testing calc_density_h20
# ------------------------------------------

def test_calc_density_h20_broadcast_failure(common_inputs):

    with pytest.raises(ValueError):
        _ = pmodel.calc_density_h2o(common_inputs.tc_mat,
                                    common_inputs.p_broadcast_err)


def test_calc_density_h20_scalars(common_inputs):

    ret = pmodel.calc_density_h2o(common_inputs.tc, common_inputs.p)
    assert round(ret, 4) == 998.2056


def test_calc_density_h20_scalar_array(common_inputs):

    ret = pmodel.calc_density_h2o(common_inputs.tc, common_inputs.p_mat)
    assert np.allclose(ret, np.array([[998.2052, 998.2056],
                                      [998.2061, 998.2066]]))


def test_calc_density_h20_arrays(common_inputs):

    ret = pmodel.calc_density_h2o(common_inputs.tc_mat, common_inputs.p_mat)
    assert np.allclose(ret, np.array([[999.1006, 998.2056],
                                      [997.0475, 995.6515]]))

