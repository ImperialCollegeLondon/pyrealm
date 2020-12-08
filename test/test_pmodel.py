import pytest
import numpy as np
import dotmap

from pypmodel import pmodel
from pypmodel.params import PARAM

# ------------------------------------------
# Fixtures: shared input values
# TODO - this could be parameterised more - many use the same signatures
#        of tc and patm as inputs.
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
                         p_broadcast_err=np.array([[100325, 101325, 103325]]),
                         soilm=0.2,
                         soilm_mat=np.array([[0.1, 0.2], [0.5, 0.7]]),
                         soilm_broadcast_err=np.array([[0.1, 0.5, 0.7]]),
                         meanalpha=1,
                         meanalpha_mat=np.array([[0.2, 1.0], [0.5, 0.7]]),
                         meanalpha_broadcast_err=np.array([[0.2, 1.0, 0.7]]),
                         elev=1000,
                         elev_mat=np.array([[900, 1000], [1100, 1200]]),
                         co2=413.03,
                         co2_mat=np.array([[373.03, 393.03], [413.03, 433.03]])
                         )

# ------------------------------------------
# Testing calc_density_h20 - temp + patm
# ------------------------------------------

def test_calc_density_h2o_broadcast_failure(common_inputs):

    with pytest.raises(ValueError):
        _ = pmodel.calc_density_h2o(common_inputs.tc_mat,
                                    common_inputs.p_broadcast_err)


def test_calc_density_h2o_scalars(common_inputs):

    ret = pmodel.calc_density_h2o(common_inputs.tc, common_inputs.p)
    assert round(ret, 4) == 998.2056


def test_calc_density_h2o_scalar_array(common_inputs):

    ret = pmodel.calc_density_h2o(common_inputs.tc, common_inputs.p_mat)
    assert np.allclose(ret, np.array([[998.2052, 998.2056],
                                      [998.2061, 998.2066]]))


def test_calc_density_h2o_arrays(common_inputs):

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

# ------------------------------------------
# Testing calc_ftemp_inst_vcmax - temp only
# ------------------------------------------


def test_calc_ftemp_kphio_scalar_c3(common_inputs):

    ret = pmodel.calc_ftemp_kphio(common_inputs.tc, c4=False)
    assert round(ret, 4) == 0.656


def test_calc_ftemp_kphio_array_c3(common_inputs):

    ret = pmodel.calc_ftemp_kphio(common_inputs.tc_mat, c4=False)
    assert np.allclose(ret, np.array([[0.6055, 0.656],
                                      [0.6895, 0.706]]))


def test_calc_ftemp_kphio_scalar_c4(common_inputs):

    ret = pmodel.calc_ftemp_kphio(common_inputs.tc, c4=True)
    assert round(ret, 4) == 0.0438


def test_calc_ftemp_kphio_array_c4(common_inputs):

    ret = pmodel.calc_ftemp_kphio(common_inputs.tc_mat, c4=True)
    assert np.allclose(ret, np.array([[0.0352, 0.0438],
                                      [0.0495, 0.0523]]))

# ------------------------------------------
# Testing calc_gammastar - temp + patm
# ------------------------------------------


def test_calc_gammastar_broadcast_failure(common_inputs):

    with pytest.raises(ValueError):
        _ = pmodel.calc_gammastar(common_inputs.tc_mat,
                                    common_inputs.p_broadcast_err)


def test_calc_gammastar_scalars(common_inputs):

    ret = pmodel.calc_gammastar(common_inputs.tc, common_inputs.p)
    assert round(ret, 4) == 3.3393


def test_calc_gammastar_scalar_array(common_inputs):

    ret = pmodel.calc_gammastar(common_inputs.tc_mat, common_inputs.p)
    assert np.allclose(ret, np.array([[2.5508606, 3.3392509],
                                      [4.3320000, 5.5718448]]))


def test_calc_gammastar_arrays(common_inputs):

    ret = pmodel.calc_gammastar(common_inputs.tc_mat, common_inputs.p_mat)
    assert np.allclose(ret, np.array([[2.5256856, 3.3392509],
                                      [4.3747535, 5.6818245]]))

# ------------------------------------------
# Testing calc_kmm - temp + patm
# ------------------------------------------


def test_calc_kmm_broadcast_failure(common_inputs):

    with pytest.raises(ValueError):
        _ = pmodel.calc_kmm(common_inputs.tc_mat,
                            common_inputs.p_broadcast_err)


def test_calc_kmm_scalars(common_inputs):

    ret = pmodel.calc_kmm(common_inputs.tc, common_inputs.p)
    assert round(ret, 6) == 46.099278


def test_calc_kmm_scalar_array(common_inputs):

    ret = pmodel.calc_kmm(common_inputs.tc_mat, common_inputs.p)
    assert np.allclose(ret, np.array([[30.044262,  46.099278],
                                      [70.842252, 108.914368]]))


def test_calc_kmm_arrays(common_inputs):

    ret = pmodel.calc_kmm(common_inputs.tc_mat, common_inputs.p_mat)
    assert np.allclose(ret, np.array([[29.877494,  46.099278],
                                      [71.146937, 109.725844]]))


# ------------------------------------------
# Testing calc_soilmstress - soilm + meanalpha
# ------------------------------------------


def test_calc_soilmstress_broadcast_failure(common_inputs):

    with pytest.raises(ValueError):
        _ = pmodel.calc_soilmstress(common_inputs.soilm_mat,
                            common_inputs.meanalpha_broadcast_err)


def test_calc_soilmstress_scalars(common_inputs):

    ret = pmodel.calc_soilmstress(common_inputs.soilm, common_inputs.meanalpha)
    assert round(ret, 6) == 0.86


def test_calc_soilmstress_scalar_array(common_inputs):

    ret = pmodel.calc_soilmstress(common_inputs.soilm_mat, common_inputs.meanalpha)
    assert np.allclose(ret, np.array([[0.78125, 0.86],
                                      [0.99125, 1.00]]))


def test_calc_soilmstress_arrays(common_inputs):

    ret = pmodel.calc_soilmstress(common_inputs.soilm_mat, common_inputs.meanalpha_mat)
    assert np.allclose(ret, np.array([[0.40069444, 0.86],
                                      [0.98173611, 1.00]]))

# ------------------------------------------
# Testing calc_viscosity_h2o - temp + patm
# ------------------------------------------


def test_calc_viscosity_h2o_broadcast_failure(common_inputs):

    with pytest.raises(ValueError):
        _ = pmodel.calc_viscosity_h2o(common_inputs.tc_mat,
                            common_inputs.p_broadcast_err)


def test_calc_viscosity_h2o_scalars(common_inputs):

    ret = pmodel.calc_viscosity_h2o(common_inputs.tc, common_inputs.p)
    assert round(ret, 7) == 0.00100160


def test_calc_viscosity_h2o_scalar_array(common_inputs):

    ret = pmodel.calc_viscosity_h2o(common_inputs.tc, common_inputs.p_mat)
    assert np.allclose(ret, np.array([[0.0010015975, 0.0010015972],
                                      [0.0010015968, 0.0010015965]]))


def test_calc_viscosity_h2o_arrays(common_inputs):

    ret = pmodel.calc_viscosity_h2o(common_inputs.tc_mat, common_inputs.p_mat)
    assert np.allclose(ret, np.array([[0.00113756998, 0.00100159716],
                                      [0.00089002254, 0.00079722171]]))



# ------------------------------------------
# Testing calc_patm - elev only
# ------------------------------------------


def test_calc_patm_scalar(common_inputs):

    ret = pmodel.calc_patm(common_inputs.elev)
    assert round(ret, 3) == 90241.542


def test_calc_patm_array(common_inputs):

    ret = pmodel.calc_patm(common_inputs.elev_mat)
    assert np.allclose(ret, np.array([[91303.561, 90241.542],
                                      [89189.548, 88147.507]]))


# ------------------------------------------
# Testing calc_co2_to_ca - co2 + patm
# ------------------------------------------


def test_calc_co2_to_ca_broadcast_failure(common_inputs):

    with pytest.raises(ValueError):
        _ = pmodel.calc_co2_to_ca(common_inputs.co2_mat,
                                  common_inputs.p_broadcast_err)


def test_calc_co2_to_ca_scalars(common_inputs):

    ret = pmodel.calc_co2_to_ca(common_inputs.co2, common_inputs.p)
    assert round(ret, 6) == 41.850265


def test_calc_co2_to_ca_scalar_array(common_inputs):

    ret = pmodel.calc_co2_to_ca(common_inputs.co2_mat, common_inputs.p)
    assert np.allclose(ret, np.array([[37.797265, 39.823765],
                                      [41.850265, 43.876765]]))


def test_calc_co2_to_ca_arrays(common_inputs):

    ret = pmodel.calc_co2_to_ca(common_inputs.co2_mat, common_inputs.p_mat)
    assert np.allclose(ret, np.array([[37.424235, 39.823765],
                                      [42.263295, 44.742825]]))

