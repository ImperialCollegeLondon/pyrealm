"""Some incredibly basic tests of P model functions.

These are primarily to help catch gross errors in the outputs during function
refactoring.
"""  # D210, D415

import numpy as np
import pytest


def test_calc_ftemp_arrh(tk=np.array([300]), ha=40000):
    """Test calc_ftemp_arrh."""
    from pyrealm.pmodel.functions import calc_ftemp_arrh

    assert np.allclose(calc_ftemp_arrh(tk, ha), 1.104622629609139)


def test_calc_ftemp_inst_rd(tc=np.array([30.0])):
    """Test calc_ftemp_inst_rd."""
    from pyrealm.pmodel.functions import calc_ftemp_inst_rd

    assert np.allclose(calc_ftemp_inst_rd(tc), 1.4455646406287255)


def test_calc_ftemp_inst_vcmax(tc=np.array([30.0])):
    """Test calc_ftemp_inst_rd."""
    from pyrealm.pmodel.functions import calc_ftemp_inst_vcmax

    assert np.allclose(calc_ftemp_inst_vcmax(tc), 1.5427221126407435)


@pytest.mark.parametrize(
    argnames="tc,c4,expected",
    argvalues=[
        (np.array([30]), False, 0.706),
        (np.array([30]), True, 0.4183999999999998),
    ],
)
def test_calc_ftemp_kphio(tc, c4, expected):
    """Test calc_ftemp_inst_rd."""
    from pyrealm.pmodel.functions import calc_ftemp_kphio

    assert np.allclose(calc_ftemp_kphio(tc, c4), expected)


def test_calc_gammastar(tc=np.array([30.0]), patm=np.array([123456])):
    """Test calc_ftemp_inst_rd."""
    from pyrealm.pmodel.functions import calc_gammastar

    assert np.allclose(calc_gammastar(tc, patm), 6.7888247955597)


def test_calc_ns_star(tc=np.array([30.0]), patm=np.array([123456])):
    """Test calc_ns_star."""
    from pyrealm.pmodel.functions import calc_ns_star

    assert np.allclose(calc_ns_star(tc, patm), 0.8957314409463492)


def test_calc_kmm(tc=np.array([30.0]), patm=np.array([123456])):
    """Test calc_kmm."""
    from pyrealm.pmodel.functions import calc_kmm

    assert np.allclose(calc_kmm(tc, patm), 117.8937532160903)


def test_calc_soilmstress_stocker(soilm=np.array([0.3])):
    """Test calc_soilmstress_stocker."""
    from pyrealm.pmodel.functions import calc_soilmstress_stocker

    assert np.allclose(calc_soilmstress_stocker(soilm), 0.93325)


def test_calc_soilmstress_mengoli(soilm=np.array([0.3])):
    """Test calc_soilmstress_mengoli."""
    from pyrealm.pmodel.functions import calc_soilmstress_mengoli

    assert np.allclose(calc_soilmstress_mengoli(soilm), 0.54705882)


def test_calc_co2_to_ca(co2=np.array([400]), patm=np.array([123456])):
    """Test calc_co2_to_ca."""
    from pyrealm.pmodel.functions import calc_co2_to_ca

    assert np.allclose(calc_co2_to_ca(co2, patm), 49.3824)
