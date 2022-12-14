import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def bigleaf():

    path = Path(__file__).parent
    with open(path / "bigleaf_test_values.json") as tvals:
        return json.load(tvals)


TEMP = np.linspace(0, 30, 31)
VP = np.linspace(1, 200, 31)
RH = np.linspace(0, 1, 31)
SH = np.linspace(0.01, 0.05, 31)
PATM = np.linspace(60, 110, 31)


@pytest.mark.parametrize(
    argnames="formula", argvalues=["Allen1998", "Alduchov1996", "Sonntag1990"]
)
def test_calc_vp_sat(bigleaf, formula):

    from pyrealm.param_classes import HygroParams
    from pyrealm.utilities import calc_vp_sat

    params = HygroParams(magnus_option=formula)
    results = calc_vp_sat(TEMP, params)
    expected = np.array(bigleaf["calc_vp_sat"][formula])

    assert np.allclose(results, expected)


@pytest.mark.parametrize(
    argnames="formula", argvalues=["Allen1998", "Alduchov1996", "Sonntag1990"]
)
def test_convert_vp_to_vpd(bigleaf, formula):

    from pyrealm.param_classes import HygroParams
    from pyrealm.utilities import convert_vp_to_vpd

    params = HygroParams(magnus_option=formula)
    results = convert_vp_to_vpd(VP, TEMP, params)
    expected = np.array(bigleaf["convert_vp_to_vpd"][formula])

    assert np.allclose(results, expected)


@pytest.mark.parametrize(
    argnames="formula", argvalues=["Allen1998", "Alduchov1996", "Sonntag1990"]
)
def test_convert_rh_to_vpd(bigleaf, formula):

    from pyrealm.param_classes import HygroParams
    from pyrealm.utilities import convert_rh_to_vpd

    params = HygroParams(magnus_option=formula)
    results = convert_rh_to_vpd(RH, TEMP, params)
    expected = np.array(bigleaf["convert_rh_to_vpd"][formula])

    assert np.allclose(results, expected)


def test_convert_sh_to_vp(bigleaf):

    from pyrealm.utilities import convert_sh_to_vp

    results = convert_sh_to_vp(SH, PATM)
    expected = np.array(bigleaf["convert_sh_to_vp"])

    assert np.allclose(results, expected)


@pytest.mark.parametrize(
    argnames="formula", argvalues=["Allen1998", "Alduchov1996", "Sonntag1990"]
)
def test_convert_sh_to_vpd(bigleaf, formula):

    from pyrealm.param_classes import HygroParams
    from pyrealm.utilities import convert_sh_to_vpd

    params = HygroParams(magnus_option=formula)
    results = convert_sh_to_vpd(SH, TEMP, PATM, params)
    expected = np.array(bigleaf["convert_sh_to_vpd"][formula])

    assert np.allclose(results, expected)
