import pytest
import numpy as np
import dotmap

from pypmodel import pmodel
from pypmodel.params import PARAM

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
# Testing calc_density_h20 - temp + patm
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

# ------------------------------------------
# Testing calc_ftemp_inst_rd - temp only but in Kelvin!
# ------------------------------------------


def test_calc_ftemp_arrh_scalar(common_inputs):

    ret = pmodel.calc_ftemp_arrh(common_inputs.tc + PARAM.k.CtoK,
                                 PARAM.KattgeKnorr.Ha)
    assert round(ret, 4) == 0.6114


def test_calc_ftemp_arrh_array(common_inputs):

    ret = pmodel.calc_ftemp_arrh(common_inputs.tc_mat+ PARAM.k.CtoK,
                                 PARAM.KattgeKnorr.Ha)
    assert np.allclose(ret, np.array([[0.367459, 0.611382],
                                      [1.000, 1.609305]]))

# ------------------------------------------
# Testing calc_ftemp_inst_vcmax - temp only
# ------------------------------------------


def test_calc_ftemp_inst_vcmax_scalar(common_inputs):

    ret = pmodel.calc_ftemp_inst_vcmax(common_inputs.tc)
    assert round(ret, 4) == 0.6371


def test_calc_ftemp_inst_vcmax_array(common_inputs):

    ret = pmodel.calc_ftemp_inst_vcmax(common_inputs.tc_mat)
    assert np.allclose(ret, np.array([[0.404673462, 0.6370757237],
                                      [1.000, 1.5427221126]]))



