"""Tests the implementation of the FastSlowModel against the reference benchmark."""

from importlib import resources

import numpy as np
import pandas
import pytest

from pyrealm.pmodel.optimal_chi import OPTIMAL_CHI_CLASS_REGISTRY


@pytest.fixture(scope="module")
def be_vie_data():
    """Import the benchmark test data."""

    # Load the BE-Vie data
    data_path = (
        resources.files("pyrealm_build_data.subdaily") / "subdaily_BE_Vie_2014.csv"
    )

    data = pandas.read_csv(str(data_path))
    data["time"] = pandas.to_datetime(data["time"])

    return data


@pytest.fixture(scope="module")
def be_vie_data_components(be_vie_data):
    """Convert the test data into a PModelEnv and arrays."""

    from pyrealm.pmodel import PModelEnvironment

    # Extract the key half hourly timestep variables
    ppfd_subdaily = be_vie_data["ppfd"].to_numpy()
    fapar_subdaily = be_vie_data["fapar"].to_numpy()
    datetime_subdaily = be_vie_data["time"].to_numpy()
    expected_gpp = be_vie_data["GPP_JAMES"].to_numpy()

    # Create the environment including some randomly distributed water variables to test
    # the methods that require those variables
    rng = np.random.default_rng()
    subdaily_env = PModelEnvironment(
        tc=be_vie_data["ta"].to_numpy(),
        vpd=be_vie_data["vpd"].to_numpy(),
        co2=be_vie_data["co2"].to_numpy(),
        patm=be_vie_data["patm"].to_numpy(),
        theta=rng.uniform(low=0.7, high=1.0, size=be_vie_data["ppfd"].shape),
        rootzonestress=rng.uniform(low=0.7, high=1.0, size=be_vie_data["ppfd"].shape),
    )

    return subdaily_env, ppfd_subdaily, fapar_subdaily, datetime_subdaily, expected_gpp


@pytest.fixture(scope="function")
def be_vie_data_components_padded(request, be_vie_data):
    """Convert the test data into a PModelEnv and arrays.

    This fixture expects calling tests to use indirect paramaterisation to provide a
    padding 2 tuple of integers, which are used to simulate incomplete days. The padding
    adds the requested number of half hourly time steps onto the original datetimes - so
    (12, 12) would extend the datetimes to from 6pm on the day before the first date to
    6am on the day after the last date.

    The actual data variables are padded with `np.nan`, as is the expected GPP, to
    preserve the expected sequence of GPP values.
    """

    from pyrealm.pmodel import PModelEnvironment

    padding = request.param

    # Pad the datetimes with actual time sequences
    datetime_subdaily = be_vie_data["time"].to_numpy()
    spacing = np.diff(datetime_subdaily)[0]

    pad_start = datetime_subdaily[0] - np.arange(padding[0], 0, -1) * spacing
    pad_end = datetime_subdaily[-1] + np.arange(1, padding[1] + 1, 1) * spacing

    datetime_subdaily = np.concatenate([pad_start, datetime_subdaily, pad_end])

    def _pad_pd_to_np(vals):
        """Add any required padding to the start and end of the data."""
        return np.pad(vals.to_numpy(), padding, constant_values=np.nan)

    ppfd_subdaily = _pad_pd_to_np(be_vie_data["ppfd"])
    fapar_subdaily = _pad_pd_to_np(be_vie_data["fapar"])
    expected_gpp = _pad_pd_to_np(be_vie_data["GPP_JAMES"])

    # Create the environment including some randomly distributed water variables to test
    # the methods that require those variables
    rng = np.random.default_rng()
    subdaily_env = PModelEnvironment(
        tc=_pad_pd_to_np(be_vie_data["ta"]),
        vpd=_pad_pd_to_np(be_vie_data["vpd"]),
        co2=_pad_pd_to_np(be_vie_data["co2"]),
        patm=_pad_pd_to_np(be_vie_data["patm"]),
        theta=rng.uniform(low=0.5, high=0.8, size=ppfd_subdaily.shape),
        rootzonestress=rng.uniform(low=0.7, high=1.0, size=ppfd_subdaily.shape),
    )

    return subdaily_env, ppfd_subdaily, fapar_subdaily, datetime_subdaily, expected_gpp


def test_FSPModel_JAMES(be_vie_data_components):
    """Test FastSlowPModel_JAMES.

    This tests the legacy calculations from the Mengoli et al JAMES paper, using that
    version of the weighted average calculations without acclimating xi.
    """

    from pyrealm.pmodel import FastSlowScaler
    from pyrealm.pmodel.subdaily import FastSlowPModel_JAMES

    env, ppfd, fapar, datetime, expected_gpp = be_vie_data_components

    # Get the fast slow scaler and set window
    fsscaler = FastSlowScaler(datetime)
    fsscaler.set_window(
        window_center=np.timedelta64(12, "h"),
        half_width=np.timedelta64(30, "m"),
    )

    # Alternate scalar used to duplicate VPD settings in JAMES implementation
    vpdscaler = FastSlowScaler(datetime)
    vpdscaler.set_nearest(time=np.timedelta64(12, "h"))

    # Fast slow model without acclimating xi with best fit adaptations to the original
    # - VPD in daily optimum using different window
    # - Jmax and Vcmax filling from midday not window end
    fs_pmodel_james = FastSlowPModel_JAMES(
        env=env,
        fs_scaler=fsscaler,
        handle_nan=True,
        kphio=1 / 8,
        fapar=fapar,
        ppfd=ppfd,
        vpd_scaler=vpdscaler,
        fill_from=np.timedelta64(12, "h"),
    )

    valid = np.logical_not(
        np.logical_or(np.isnan(expected_gpp), np.isnan(fs_pmodel_james.gpp))
    )

    # Test that non-NaN predictions are within 0.5% - slight differences in constants
    # and rounding of outputs prevent a closer match between the implementations
    assert np.allclose(
        fs_pmodel_james.gpp[valid],
        expected_gpp[valid] * env.core_const.k_c_molmass,
        rtol=0.005,
    )


def test_FSPModel_corr(be_vie_data_components):
    """Test FastSlowPModel.

    This tests that the pyrealm implementation including acclimating xi at least
    correlates well with the legacy calculations from the Mengoli et al JAMES paper
    without acclimating xi.
    """

    from pyrealm.pmodel import FastSlowScaler, PModel
    from pyrealm.pmodel.new_subdaily import SubdailyPModel

    env, ppfd, fapar, datetime, expected_gpp = be_vie_data_components

    # Fit the standard P Model
    pmodel = PModel(env=env, kphio=1 / 8)
    pmodel.estimate_productivity(fapar=fapar, ppfd=ppfd)

    # Get the fast slow scaler and set window
    fsscaler = FastSlowScaler(datetime)
    fsscaler.set_window(
        window_center=np.timedelta64(12, "h"),
        half_width=np.timedelta64(30, "m"),
    )

    # Run as a subdaily model
    fs_pmodel = SubdailyPModel(
        env=env,
        ppfd=ppfd,
        fapar=fapar,
        fs_scaler=fsscaler,
        handle_nan=True,
    )

    valid = np.logical_not(
        np.logical_or(np.isnan(expected_gpp), np.isnan(fs_pmodel.gpp))
    )

    # Test that non-NaN predictions correlate well and are approximately the same
    gpp_in_micromols = fs_pmodel.gpp[valid] / env.core_const.k_c_molmass
    assert np.allclose(gpp_in_micromols, expected_gpp[valid], rtol=0.2)
    r_vals = np.corrcoef(gpp_in_micromols, expected_gpp[valid])
    assert np.all(r_vals > 0.995)


@pytest.mark.parametrize(
    argnames="be_vie_data_components_padded",
    argvalues=[
        pytest.param((0, 0), id="no pad"),
        pytest.param((12, 0), id="start pad"),
        pytest.param((0, 12), id="end pad"),
        pytest.param((12, 12), id="pad both"),
    ],
    indirect=["be_vie_data_components_padded"],
)
def test_FSPModel_corr_padded(be_vie_data_components_padded):
    """Test FastSlowPModel.

    This tests that the pyrealm implementation including acclimating xi at least
    correlates well with the legacy calculations from the Mengoli et al JAMES paper
    without acclimating xi.

    The fixture providing the data uses indirect parameterisation to pad the data to
    have incomplete days to check that the handling of the data holds up when the dates
    are changed. Note that this does nothing at all to the calculations - the data are
    padded with np.nan - so this is mostly checking that padding by itself does not
    raise issues.
    """

    from pyrealm.pmodel import FastSlowScaler, PModel
    from pyrealm.pmodel.new_subdaily import SubdailyPModel

    env, ppfd, fapar, datetime, expected_gpp = be_vie_data_components_padded

    # Fit the standard P Model
    pmodel = PModel(env=env, kphio=1 / 8)
    pmodel.estimate_productivity(fapar=fapar, ppfd=ppfd)

    # Get the fast slow scaler and set window
    fsscaler = FastSlowScaler(datetime)
    fsscaler.set_window(
        window_center=np.timedelta64(12, "h"),
        half_width=np.timedelta64(30, "m"),
    )

    # Run as a subdaily model
    fs_pmodel = SubdailyPModel(
        env=env,
        ppfd=ppfd,
        fapar=fapar,
        fs_scaler=fsscaler,
        handle_nan=True,
    )

    valid = np.logical_not(
        np.logical_or(np.isnan(expected_gpp), np.isnan(fs_pmodel.gpp))
    )

    # Test that non-NaN predictions correlate well and are approximately the same
    gpp_in_micromols = fs_pmodel.gpp[valid] / env.core_const.k_c_molmass
    assert np.allclose(gpp_in_micromols, expected_gpp[valid], rtol=0.2)
    r_vals = np.corrcoef(gpp_in_micromols, expected_gpp[valid])
    assert np.all(r_vals > 0.995)


@pytest.mark.parametrize("ndims", [2, 3, 4])
def test_FSPModel_dimensionality(be_vie_data, ndims):
    """Tests that the FastSlowPModel handles dimensions correctly.

    This broadcasts the BE-Vie onto more dimensions and checks that the code iterates
    over those dimensions correctly. fAPAR and PPFD are then fixed across the other
    dimensions to check the results scale as expected.
    """

    from pyrealm.pmodel import FastSlowScaler, PModelEnvironment
    from pyrealm.pmodel.new_subdaily import SubdailyPModel

    datetime = be_vie_data["time"].to_numpy()

    # Set up the dimensionality for the test - create a shape tuple with extra
    # dimensions and then broadcast the model inputs onto it. These need to be
    # transposed to return datetime to the first axis. When n_dim = 1, the data are
    # passed as is.
    extra_dims = [3] * (ndims - 1)
    array_dims = tuple(extra_dims + [len(datetime)])

    # Create the environment
    env = PModelEnvironment(
        tc=np.broadcast_to(be_vie_data["ta"], array_dims).transpose(),
        vpd=np.broadcast_to(be_vie_data["vpd"], array_dims).transpose(),
        co2=np.broadcast_to(be_vie_data["co2"], array_dims).transpose(),
        patm=np.broadcast_to(be_vie_data["patm"], array_dims).transpose(),
    )

    # Get the fast slow scaler and set window
    fsscaler = FastSlowScaler(datetime)
    fsscaler.set_window(
        window_center=np.timedelta64(12, "h"),
        half_width=np.timedelta64(30, "m"),
    )

    # Apply a different random value of fAPAR for each time series
    fapar_vals = np.random.random(extra_dims)
    fapar_vals[0] = 1.0

    # Fast slow model
    fs_pmodel = SubdailyPModel(
        env=env,
        fs_scaler=fsscaler,
        fapar=fapar_vals * np.ones(array_dims).transpose(),
        ppfd=np.ones(array_dims).transpose(),
        handle_nan=True,
    )

    # The GPP along the timescale of the different dimensions should be directly
    # proportional to the random fapar values and hence GPP/FAPAR should all equal the
    # value when it is set to 1
    timeaxis_mean = np.nansum(fs_pmodel.gpp, axis=0)
    assert np.allclose(timeaxis_mean / fapar_vals, timeaxis_mean[0])


@pytest.mark.parametrize("method_optchi", OPTIMAL_CHI_CLASS_REGISTRY.keys())
def test_Subdaily_opt_chi_methods(be_vie_data_components, method_optchi):
    """Tests that the SubdailyModel runs with all the provided optimal chi classes.

    This currently just checks that the subdaily model _runs_ with each of the different
    implementations of the OptimalChi ABC.
    """

    from pyrealm.pmodel import FastSlowScaler
    from pyrealm.pmodel.new_subdaily import SubdailyPModel

    env, ppfd, fapar, datetime, _ = be_vie_data_components

    # Get the fast slow scaler and set window
    fsscaler = FastSlowScaler(datetime)
    fsscaler.set_window(
        window_center=np.timedelta64(12, "h"),
        half_width=np.timedelta64(30, "m"),
    )

    # Run as a subdaily model and it should complete.
    _ = SubdailyPModel(
        env=env,
        fs_scaler=fsscaler,
        fapar=fapar,
        ppfd=ppfd,
        method_optchi=method_optchi,
        handle_nan=True,
    )


@pytest.mark.parametrize("method_optchi", OPTIMAL_CHI_CLASS_REGISTRY.keys())
def test_convert_pmodel_to_subdaily(be_vie_data_components, method_optchi):
    """Tests the convert_pmodel_to_subdaily method."""

    from pyrealm.pmodel import FastSlowScaler, PModel
    from pyrealm.pmodel.new_subdaily import SubdailyPModel, convert_pmodel_to_subdaily

    env, ppfd, fapar, datetime, _ = be_vie_data_components

    # Get the fast slow scaler and set window
    fsscaler = FastSlowScaler(datetime)
    fsscaler.set_window(
        window_center=np.timedelta64(12, "h"),
        half_width=np.timedelta64(30, "m"),
    )

    # Run as a subdaily model
    direct = SubdailyPModel(
        env=env,
        fs_scaler=fsscaler,
        fapar=fapar,
        ppfd=ppfd,
        method_optchi=method_optchi,
        handle_nan=True,
    )

    # Convert a standard model
    standard_model = PModel(env=env, kphio=1 / 8, method_optchi=method_optchi)
    standard_model.estimate_productivity(fapar=fapar, ppfd=ppfd)

    converted = convert_pmodel_to_subdaily(
        pmodel=standard_model, fs_scaler=fsscaler, handle_nan=True
    )

    assert np.allclose(converted.gpp, direct.gpp, equal_nan=True)
