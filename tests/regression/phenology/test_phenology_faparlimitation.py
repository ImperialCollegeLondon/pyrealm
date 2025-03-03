"""Test the FaparLimitation class."""

from importlib import resources

import numpy as np
import pytest


@pytest.fixture()
def annual_data():
    """Load the input data from netcdf file."""

    from pyrealm_build_data.LAI_in_pyrealm import faparlim_data_io

    datafile = (
        resources.files("pyrealm_build_data") / "LAI_in_pyrealm/faparlim_input.nc"
    )

    return faparlim_data_io.read_faparlim_input(datafile)


@pytest.mark.parametrize(
    argnames="exp_faparmax, exp_laimax",
    argvalues=[
        (
            np.array(
                [
                    0.98260768,
                    0.9841239,
                    0.98288146,
                    0.98659916,
                    0.98429005,
                    0.98489114,
                    0.98490976,
                    0.98668437,
                    0.98606508,
                    0.98480568,
                    0.98675439,
                ]
            ),
            np.array(
                [
                    8.10345273,
                    8.28588143,
                    8.1351861,
                    8.62487565,
                    8.30692182,
                    8.38494796,
                    8.38741475,
                    8.63763276,
                    8.54671508,
                    8.37366785,
                    8.64817802,
                ]
            ),
        )
    ],
)
def test_faparlimitation(annual_data, exp_faparmax, exp_laimax):
    """Regression test for FaparLimitation constructor."""

    from pyrealm.phenology.fapar_limitation import FaparLimitation

    (
        annual_total_A0_subdaily,
        annual_total_P,
        aridity_index,
        annual_mean_ca,
        annual_mean_chi,
        annual_mean_vpd,
    ) = annual_data

    faparlim = FaparLimitation(
        annual_total_A0_subdaily,
        annual_mean_ca,
        annual_mean_chi,
        annual_mean_vpd,
        annual_total_P,
        aridity_index,
    )

    assert np.allclose(exp_faparmax, faparlim.fapar_max)
    assert np.allclose(exp_laimax, faparlim.lai_max)
