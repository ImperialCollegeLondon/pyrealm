"""Run tests of the PModel.

These tests compare the pyrealm implementation to the outputs of the rpmodel
implementation.
"""

import json
import warnings
from contextlib import nullcontext as does_not_raise
from importlib import resources

import numpy as np
import pytest

# flake8: noqa D103 - docstrings on unit tests

# RPMODEL bugs
# rpmodel was using an incorrect parameterisation of the C4 ftemp kphio curve
# that is fixed but currently (1.2.0) an implementation error in the output checking
# means this still have to be skipped.

RPMODEL_C4_BUG = True


# ------------------------------------------
# Fixtures: inputs and expected values
# ------------------------------------------


@pytest.fixture(scope="module")
def values():
    """Fixture to load test inputs and expected rpmodel outputs from file."""

    datapath = (
        resources.files("pyrealm_build_data.rpmodel") / "test_outputs_rpmodel.json"
    )

    with open(str(datapath)) as infile:
        values = json.load(infile)

    # JSON contains nested dictionary of scalars and lists - convert
    # the lists to ndarrays, and use float to ensure that None --> np.nan
    # rather than the default output of dtype object.

    def lists_to_ndarray(d):
        for k, v in d.items():
            if isinstance(v, dict):
                lists_to_ndarray(v)
            elif isinstance(v, list):
                d[k] = np.array(v, dtype=float)
            else:
                pass

    lists_to_ndarray(values)

    return values


# ------------------------------------------
# Testing CalcOptimalChi - vpd + internals kmm, gammastar, ns_star, ca
# ------------------------------------------


@pytest.mark.parametrize(
    "tc, patm, co2, vpd, method, context_manager, expvalues",
    [
        (
            "tc_sc",
            "patm_sc",
            "co2_sc",
            "vpd_sc",
            "c4",
            does_not_raise(),
            "optchi_p14_sc_c4",
        ),  # scalar, c4
        (
            "tc_ar",
            "patm_sc",
            "co2_sc",
            "vpd_sc",
            "c4",
            does_not_raise(),
            "optchi_p14_mx_c4",
        ),  # scalar + arrays, c4
        (
            "tc_ar",
            "patm_ar",
            "co2_ar",
            "vpd_ar",
            "c4",
            does_not_raise(),
            "optchi_p14_ar_c4",
        ),  # arrays, c4
        # (
        #     "shape_error",
        #     "patm_ar",
        #     "co2_ar",
        #     "vpd_ar",
        #     "c4",
        #     pytest.raises(ValueError),
        #     None,
        # ),  # shape error, c4
        (
            "tc_sc",
            "patm_sc",
            "co2_sc",
            "vpd_sc",
            "prentice14",
            does_not_raise(),
            "optchi_p14_sc_c3",
        ),  # scalar, c3
        (
            "tc_ar",
            "patm_sc",
            "co2_sc",
            "vpd_sc",
            "prentice14",
            does_not_raise(),
            "optchi_p14_mx_c3",
        ),  # scalar + arrays, c3
        (
            "tc_ar",
            "patm_ar",
            "co2_ar",
            "vpd_ar",
            "prentice14",
            does_not_raise(),
            "optchi_p14_ar_c3",
        ),  # arrays, c3
        # (
        #     "shape_error",
        #     "patm_ar",
        #     "co2_ar",
        #     "vpd_ar",
        #     "prentice14",
        #     pytest.raises(ValueError),
        #     None,
        # ),  # shape error, c3
    ],
)
def test_calc_optimal_chi_new(
    values, tc, patm, co2, vpd, method, context_manager, expvalues
):
    """Test the CalcOptimalChi class."""

    from pyrealm.pmodel import PModelEnvironment
    from pyrealm.pmodel.calc_optimal_chi_new import OPTIMAL_CHI_CLASS_REGISTRY

    with context_manager:
        env = PModelEnvironment(
            tc=values[tc], patm=values[patm], vpd=values[vpd], co2=values[co2]
        )

        # Retrieve the appropriate implementation from the registry
        method_class = OPTIMAL_CHI_CLASS_REGISTRY[method]
        ret = method_class(env)

        if expvalues is not None:
            expected = values[expvalues]
            assert np.allclose(ret.chi, expected["chi"])
            assert np.allclose(ret.mj, expected["mj"])
            assert np.allclose(ret.mc, expected["mc"])
            assert np.allclose(ret.mjoc, expected["mjoc"])
