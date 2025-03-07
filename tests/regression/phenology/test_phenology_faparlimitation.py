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
                    0.00741829,
                    0.00657795,
                    0.00663436,
                    0.00792586,
                    0.00744838,
                    0.00790967,
                    0.00790302,
                    0.00552947,
                    0.00616514,
                    0.00852866,
                    0.00539698,
                ]
            ),
            np.array(
                [
                    0.01489189,
                    0.01319937,
                    0.01331292,
                    0.01591488,
                    0.01495251,
                    0.01588223,
                    0.01586883,
                    0.01108962,
                    0.01236845,
                    0.01713047,
                    0.0108232,
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


@pytest.fixture()
def pmodel_data():
    """Load the input data for the from_pmodel class function from netcdf file."""

    from pyrealm_build_data.LAI_in_pyrealm import faparlim_data_io

    datafile = (
        resources.files("pyrealm_build_data")
        / "LAI_in_pyrealm/faparlim_pmodel_input.nc"
    )

    return faparlim_data_io.read_pmodel_faparlim_input(datafile)


@pytest.mark.parametrize(
    argnames="exp_faparmax, exp_laimax",
    argvalues=[
        (
            np.array(
                [
                    0.00890042,
                    0.00748109,
                    0.00538652,
                    0.00979542,
                    0.00671602,
                    0.00627187,
                    0.00778598,
                    0.00556823,
                    0.00648213,
                    0.00921882,
                    0.00805771,
                ]
            ),
            np.array(
                [
                    0.01788053,
                    0.01501843,
                    0.01080216,
                    0.01968741,
                    0.01347735,
                    0.01258324,
                    0.01563289,
                    0.01116758,
                    0.01300646,
                    0.01852314,
                    0.0161807,
                ]
            ),
        )
    ],
)
def test_faparlimitation_frompmodel(pmodel_data, exp_faparmax, exp_laimax):
    """Regression test for from_pmodel FaparLimitation class method."""

    from pyrealm.phenology.fapar_limitation import FaparLimitation
    from pyrealm.pmodel import PModel, PModelEnvironment

    (
        tc,
        vpd,
        co2,
        patm,
        growing_season,
        datetimes,
        precipitation,
        aridity_index,
        ppfd,
    ) = pmodel_data

    env = PModelEnvironment(tc=tc, vpd=vpd, co2=co2, patm=patm)

    pmodel = PModel(
        env=env,
        reference_kphio=1 / 8,
        method_kphio="temperature",
    )

    pmodel.estimate_productivity(fapar=np.ones_like(env.ca), ppfd=ppfd)

    faparlim = FaparLimitation.from_pmodel(
        pmodel, growing_season, datetimes, precipitation, aridity_index
    )

    assert np.allclose(exp_faparmax, faparlim.fapar_max)
    assert np.allclose(exp_laimax, faparlim.lai_max)
