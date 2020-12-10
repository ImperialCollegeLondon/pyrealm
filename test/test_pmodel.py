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
    
    prop_mat = np.array([[0.95, 1.05], [1.0, 1.1]])
    kmm = 46.09928
    gammastar = 3.33925
    ns_star = 1.12536
    ca = 40.53
    vpd = 1000

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
                         co2_mat=np.array([[373.03, 393.03], [413.03, 433.03]]),
                         kmm=kmm,
                         gammastar=gammastar,
                         ns_star=ns_star,
                         ca=ca,
                         vpd=vpd,
                         kmm_mat=kmm * prop_mat,
                         gammastar_mat=gammastar * prop_mat,
                         gammastar_broadcast_err=gammastar * np.array([0.95, 1.0, 1.05]),
                         ns_star_mat=ns_star * prop_mat,
                         ca_mat=ca * prop_mat,
                         vpd_mat=vpd * prop_mat
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

    ret = pmodel.calc_ftemp_arrh(common_inputs.tc_mat + PARAM.k.CtoK,
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

# ------------------------------------------
# Testing CalcOptimalChi - vpd + internals kmm, gammastar, ns_star, ca
# ------------------------------------------


def test_calc_optimal_chi_broadcast_failure(common_inputs):

    with pytest.raises(ValueError):
        _ = pmodel.CalcOptimalChi(common_inputs.kmm_mat,
                                  common_inputs.gammastar_broadcast_err,
                                  common_inputs.ns_star,
                                  common_inputs.ca,
                                  common_inputs.vpd)


# NOTE - the c4 method __INTENTIONALLY__ always returns scalars
# regardless of input shape

def test_calc_optimal_chi_c4_scalars(common_inputs):

    ret = pmodel.CalcOptimalChi(common_inputs.kmm,
                                common_inputs.gammastar,
                                common_inputs.ns_star,
                                common_inputs.ca,
                                common_inputs.vpd,
                                method='c4')

    assert ret.chi == 1.0
    assert ret.mj == 1.0
    assert ret.mjoc == 1.0
    assert ret.mc == 1.0


def test_calc_optimal_chi_c4_scalars_arrays(common_inputs):
    ret = pmodel.CalcOptimalChi(common_inputs.kmm_mat,
                                common_inputs.gammastar,
                                common_inputs.ns_star_mat,
                                common_inputs.ca,
                                common_inputs.vpd,
                                method='c4')

    assert ret.chi == 1.0
    assert ret.mj == 1.0
    assert ret.mjoc == 1.0
    assert ret.mc == 1.0


def test_calc_optimal_chi_c4_arrays(common_inputs):
    ret = pmodel.CalcOptimalChi(common_inputs.kmm_mat,
                                common_inputs.gammastar_mat,
                                common_inputs.ns_star_mat,
                                common_inputs.ca_mat,
                                common_inputs.vpd_mat,
                                method='c4')

    assert ret.chi == 1.0
    assert ret.mj == 1.0
    assert ret.mjoc == 1.0
    assert ret.mc == 1.0

# Prentice 14


def test_calc_optimal_chi_scalars(common_inputs):

    ret = pmodel.CalcOptimalChi(common_inputs.kmm,
                                common_inputs.gammastar,
                                common_inputs.ns_star,
                                common_inputs.ca,
                                common_inputs.vpd)

    assert round(ret.chi, 8) == 0.69435213
    assert round(ret.mc, 8) == 0.33408383
    assert round(ret.mj, 8) == 0.71230386
    assert round(ret.mjoc, 8) == 2.13211114


def test_calc_optimal_chi_scalars_arrays(common_inputs):
    ret = pmodel.CalcOptimalChi(common_inputs.kmm_mat,
                                common_inputs.gammastar,
                                common_inputs.ns_star_mat,
                                common_inputs.ca,
                                common_inputs.vpd)

    assert np.allclose(ret.chi, np.array([[0.69471370,  0.69402371],
                                          [0.69435213,  0.69372406]]))
    assert np.allclose(ret.mc, np.array([[0.34492189,  0.32390633],
                                         [0.33408383,  0.31433074]]))
    assert np.allclose(ret.mj, np.array([[0.71242488,  0.71219384],
                                         [0.71230386,  0.71209338]]))
    assert np.allclose(ret.mjoc, np.array([[2.0654673,  2.1987648],
                                           [2.1321111,  2.2654271]]))


def test_calc_optimal_chi_arrays(common_inputs):
    ret = pmodel.CalcOptimalChi(common_inputs.kmm_mat,
                                common_inputs.gammastar_mat,
                                common_inputs.ns_star_mat,
                                common_inputs.ca_mat,
                                common_inputs.vpd_mat)

    assert np.allclose(ret.chi, np.array([[0.69955736,  0.68935938],
                                          [0.69435213,  0.68456214]]))
    assert np.allclose(ret.mc, np.array([[0.33597077,  0.33226381],
                                         [0.33408383,  0.33050567]]))
    assert np.allclose(ret.mj, np.array([[0.71403643,  0.71062217],
                                         [0.71230386,  0.70898771]]))
    assert np.allclose(ret.mjoc, np.array([[2.1252933,  2.1387287],
                                           [2.1321111,  2.1451605]]))

