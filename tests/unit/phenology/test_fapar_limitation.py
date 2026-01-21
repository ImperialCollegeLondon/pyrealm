"""Tests the FaparLimitation class."""

import json
from contextlib import nullcontext as does_not_raise
from importlib import resources
from typing import Literal

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose


@pytest.fixture
def annual_inputs():
    """Annual inputs from the regression dataset as a dictionary of numpy arrays."""
    datapath = (
        resources.files("pyrealm_build_data.phenology.inputs.fortnightly")
        / "annual_inputs.csv"
    )

    data = pd.read_csv(str(datapath))
    data_dict = {k: v.to_numpy() for k, v in data.items()}
    data_dict["time"] = data_dict["year"].astype(str).astype("datetime64[Y]")
    return data_dict


@pytest.fixture
def cai_zhou_fapar_max():
    """Annual fapar max predictions from the Cai/Zhou method."""
    datapath = (
        resources.files("pyrealm_build_data.phenology.cai_zhou_method")
        / "fapar_max_predictions.csv"
    )

    data = pd.read_csv(str(datapath))
    data_dict = {k: v.to_numpy() for k, v in data.items()}
    data_dict["time"] = data_dict["year"].astype(str).astype("datetime64[Y]")
    return data_dict


@pytest.fixture
def fortnightly_inputs():
    """Fortnightly data from the regression dataset as a dictionary of numpy arrays."""
    datapath = (
        resources.files("pyrealm_build_data.phenology.inputs.fortnightly")
        / "pmodel_inputs.csv"
    )

    data = pd.read_csv(str(datapath))
    data_dict = {k: v.to_numpy() for k, v in data.items()}
    data_dict["time"] = data_dict["time"].astype("datetime64[s]")
    return data_dict


@pytest.fixture
def site_data():
    """Load the site data."""

    datafile = (
        resources.files("pyrealm_build_data.phenology.inputs.source")
        / "DE-GRI_site_data.json"
    )

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


def resize_aridity_index_arrays(
    ai_value,
    shape,
    mode=Literal["scalar", "constant", "full"],
):
    """Utility function to generate AI inputs with varying dimensionality.

    The modes provide different potential ways of scaling AI values for other inputs of
    differing dimensionality. Scalar uses a single value for all sites, full provides an
    observation for each observation, and constant provides a single value for each site
    (which would be the usual way of working).

    dims  scalar   full           constant
    1     (1,)     (11,)          (1,)
    2     (1,)     (11, 3)        (1, 3)
    3     (1,)     (11, 3, 3)     (1, 3, 3)
    4     (1,)     (11, 3, 3, 3)  (1, 3, 3, 3)
    """

    if mode == "scalar":
        # Scalar array - return a 1D
        shape = (1,)

    if mode == "constant":
        # Constant across time - return an array that has a singleton dimension on first
        # axis
        shape = (1, *shape[1:])

    # Otherwise the full shape is returned
    return np.broadcast_to(ai_value, shape)


@pytest.mark.parametrize(argnames="ai_mode", argvalues=("scalar", "constant", "full"))
@pytest.mark.parametrize(argnames="extra_dims", argvalues=(0, 1, 2, 3))
def test_FaparLimitation_dimensionality(
    site_data, annual_inputs, cai_zhou_fapar_max, ai_mode, extra_dims
):
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
        ("annual_total_potential_gpp", "annual_total_A0"),
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
        cai_zhou_fapar_max["fapar_max_ft"][:, *([np.newaxis] * extra_dims)],
        target_shape,
    )

    ai_inputs = resize_aridity_index_arrays(
        site_data["AI"], shape=target_shape, mode=ai_mode
    )
    # Scalar AI value
    faparlim = FaparLimitation(
        years=annual_inputs["time"],
        aridity_index=ai_inputs,
        **array_inputs,
    )

    assert_allclose(faparlim.fapar_max, expected_faparmax)


@pytest.mark.parametrize(argnames="ai_mode", argvalues=("scalar", "constant", "full"))
@pytest.mark.parametrize(argnames="extra_dims", argvalues=(0, 1, 2, 3))
def test_FaparLimitation_from_pmodel(
    site_data, fortnightly_inputs, cai_zhou_fapar_max, ai_mode, extra_dims
):
    """Test FaparLimitation works with different input dimensionality.

    This test also checks that various dimensions of the aridity index also work as
    intended.
    """
    from pyrealm.phenology.fapar_limitation import FaparLimitation
    from pyrealm.pmodel import PModel, PModelEnvironment

    # Set up the dimensionality for the test - create a shape tuple with extra
    # dimensions and then broadcast the model inputs onto it. When extra_dims = 0, the
    # data are passed as is.
    n_obs = len(fortnightly_inputs["time"])
    n_years = len(cai_zhou_fapar_max["time"])

    # Create a target shape with the values along the first axis repeated 3 times along
    # each of the new dimensions
    target_shape = tuple([n_obs, *([3] * extra_dims)])
    target_shape_expected = tuple([n_years, *([3] * extra_dims)])

    # The code below takes the argument names and input data names and then:
    # - adds extra dimensions to the 1D inputs: e.g. (11,) -> (11, 1, 1)
    # - broadcasts that data to the target shape: e.g. (11, 1, 1) -> (11, 3, 3)
    array_inputs = {
        key: np.broadcast_to(
            val[:, *([np.newaxis] * extra_dims)],
            target_shape,
        )
        for key, val in fortnightly_inputs.items()
    }

    # Pop off the variables not needed for PModelEnvironment
    precip = array_inputs.pop("precip_molar")
    growing_season = array_inputs.pop("growing_season")

    # Calculate the PModel
    pmodel_env = PModelEnvironment(
        **array_inputs,
    )
    pmodel = PModel(pmodel_env)

    ai_inputs = resize_aridity_index_arrays(
        site_data["AI"], shape=target_shape_expected, mode=ai_mode
    )

    # Scalar AI value
    faparlim = FaparLimitation.from_pmodel(
        pmodel=pmodel,
        datetimes=fortnightly_inputs["time"],
        aridity_index=ai_inputs,
        precip=precip,
        growing_season=growing_season,
    )

    # Expected values
    expected_faparmax = np.broadcast_to(
        cai_zhou_fapar_max["fapar_max_ft"][:, *([np.newaxis] * extra_dims)],
        target_shape_expected,
    )
    assert_allclose(faparlim.fapar_max, expected_faparmax)
