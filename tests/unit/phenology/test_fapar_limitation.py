"""Tests the FaparLimitation class."""

import json
from contextlib import nullcontext as does_not_raise
from importlib import resources

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose


@pytest.fixture
def annual_inputs():
    """Annual inputs from the regression dataset as a dictionary of numpy arrays."""
    datapath = (
        resources.files("pyrealm_build_data.phenology.fortnightly_example")
        / "annual_outputs.csv"
    )

    data = pd.read_csv(str(datapath))
    data_dict = {k: v.to_numpy() for k, v in data.items()}
    data_dict["time"] = data_dict["time"].astype("datetime64[Y]")
    return data_dict


@pytest.fixture
def site_data():
    """Load the site data."""

    datafile = resources.files("pyrealm_build_data") / "phenology/DE-GRI_site_data.json"

    with open(str(datafile)) as json_src:
        site_data = json.load(json_src)

    return site_data


@pytest.mark.parametrize(
    argnames="aridity_index, raises",
    argvalues=[
        (0, pytest.raises(ValueError)),
        (-1, pytest.raises(ValueError)),
        (1, does_not_raise()),
        (np.array([1, 1, 1]), does_not_raise()),
        (np.array([1, 1, -1]), pytest.raises(ValueError)),
    ],
)
def test_aridity_index_check(aridity_index, raises):
    """Tests if the AI positivity check is works for both scalar and vector inputs."""
    from pyrealm.phenology.fapar_limitation import FaparLimitation

    # Inputs are arrays of length 3
    shape = (3,)
    with raises:
        FaparLimitation(
            annual_total_potential_gpp=np.ones(shape),
            annual_mean_ca=np.ones(shape),
            annual_mean_chi=np.ones(shape),
            annual_mean_vpd=np.ones(shape),
            annual_total_precip=np.ones(shape),
            annual_growing_season_length=np.ones(shape),
            years=np.repeat(np.datetime64("2000"), shape),
            aridity_index=aridity_index,
        )


@pytest.mark.parametrize(argnames="extra_dims", argvalues=(0, 1, 2, 3))
def test_FaparLimitation_dimensionality(site_data, annual_inputs, extra_dims):
    """Test FaparLimitation works with different input dimensionality.

    This test also checks that various dimensions of the aridity index also work as
    intended.
    """
    from pyrealm.phenology.fapar_limitation import FaparLimitation

    # Set up the dimensionality for the test - create a shape tuple with extra
    # dimensions and then broadcast the model inputs onto it. When extra_dims = 0, the
    # data are passed as is.
    n_years = len(annual_inputs["time"])

    # Create a target shape with the values along the first axis repeated 3 times along
    # each of the new dimensions
    target_shape = tuple([n_years, *([3] * extra_dims)])

    array_input_vars = (
        ("annual_total_potential_gpp", "ann_total_A0"),
        ("annual_mean_ca", "annual_mean_ca_in_GS"),
        ("annual_mean_chi", "annual_mean_chi_in_GS"),
        ("annual_mean_vpd", "annual_mean_VPD_in_GS"),
        ("annual_total_precip", "annual_precip_molar"),
        ("annual_growing_season_length", "N_growing_days"),
    )

    # The code below takes the argument names and input data names and then:
    # - adds extra dimensions to the 1D inputs: e.g. (11,) -> (11, 1, 1)
    # - broadcasts that data to the target shape: e.g. (11, 1, 1) -> (11, 3, 3)
    array_inputs = {
        arg_name: np.broadcast_to(
            annual_inputs[data_name][:, *([np.newaxis] * extra_dims)],
            target_shape,
        )
        for arg_name, data_name in array_input_vars
    }

    # Expected values
    expected_faparmax = np.broadcast_to(
        annual_inputs["fapar_max"][:, *([np.newaxis] * extra_dims)],
        target_shape,
    )

    # Scalar AI value
    faparlim = FaparLimitation(
        years=annual_inputs["time"],
        aridity_index=site_data["AI"],
        **array_inputs,
    )

    assert_allclose(faparlim.fapar_max, expected_faparmax)

    # Full rank AI - repeating values across years.
    faparlim_ai_fullsize = FaparLimitation(
        years=annual_inputs["time"],
        aridity_index=np.broadcast_to(
            site_data["AI"],
            target_shape,
        ),
        **array_inputs,
    )

    assert_allclose(faparlim_ai_fullsize.fapar_max, expected_faparmax)

    # AI matches shape of non-year dimensions - held constant across years - this
    # creates AI arrays with target_shape, ai_shape pairs as below to comply with
    # broadcasting rules.
    # (11,) (1,)
    # (11, 3) (1, 3)
    # (11, 3, 3) (1, 3, 3)
    # (11, 3, 3, 3) (1, 3, 3, 3)
    ai_constant = np.broadcast_to(site_data["AI"], tuple([3] * extra_dims))
    ai_constant = ai_constant[np.newaxis, ...]

    faparlim_ai_constant = FaparLimitation(
        years=annual_inputs["time"],
        aridity_index=ai_constant,
        **array_inputs,
    )

    assert_allclose(faparlim_ai_constant.fapar_max, expected_faparmax)
