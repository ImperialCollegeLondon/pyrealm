"""Some incredibly basic tests of P model functions.

These are primarily to help catch gross errors in the outputs during function
refactoring.
"""  # D210, D415

import numpy as np
from numpy.testing import assert_allclose

from pyrealm.constants.core_const import CoreConst

TC = np.array([30.0])
core_const = CoreConst()
TK = TC + core_const.k_CtoK
PATM = np.array([123456])


def test_calculate_ftemp_inst_rd(tc=TC):
    """Test calculate_ftemp_inst_rd."""

    from pyrealm.pmodel.functions import calculate_ftemp_inst_rd

    assert_allclose(
        calculate_ftemp_inst_rd(tc=tc),
        1.4455646406287255,
    )


def test_calculate_gammastar(tk=TK, patm=PATM):
    """Test calculate_gammastar."""

    from pyrealm.pmodel.functions import calculate_gammastar

    assert_allclose(
        calculate_gammastar(tk, patm),
        6.7888247955597,
    )


def test_calculate_ns_star(tk=TK, patm=PATM):
    """Test calculate_ns_star."""
    from pyrealm.pmodel.functions import calculate_ns_star

    assert_allclose(calculate_ns_star(tk, patm), 0.8950389362238037)


def test_calculate_kmm(tk=TK, patm=PATM):
    """Test calculate_kmm."""
    from pyrealm.pmodel.functions import calculate_kmm

    assert_allclose(calculate_kmm(tk, patm), 117.8937532160903)


def test_calculate_soilmstress_stocker(soilm=np.array([0.3])):
    """Test calculate_soilmstress_stocker."""
    from pyrealm.pmodel.functions import calculate_soilmstress_stocker

    assert_allclose(calculate_soilmstress_stocker(soilm), 0.93325)


def test_calculate_soilmstress_mengoli(soilm=np.array([0.3])):
    """Test calculate_soilmstress_mengoli."""
    from pyrealm.pmodel.functions import calculate_soilmstress_mengoli

    assert_allclose(calculate_soilmstress_mengoli(soilm), 0.54705882)


def test_calculate_co2_to_ca(co2=np.array([400]), patm=PATM):
    """Test calculate_co2_to_ca."""
    from pyrealm.pmodel.functions import calculate_co2_to_ca

    assert_allclose(calculate_co2_to_ca(co2, patm), 49.3824)
