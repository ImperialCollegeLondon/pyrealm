from importlib import resources

import numpy as np
import pytest


@pytest.fixture(scope="module")
def be_vie_data():
    """Import the test data"""

    # This feels like a hack but it isn't obvious how to reference the data files
    # included in the source distribution from the package path.
    data_path = resources.files("pyrealm_build_data") / "subdaily_BE_Vie_2014.csv"

    data = np.genfromtxt(
        data_path,
        names=True,
        delimiter=",",
        dtype=None,
        encoding="UTF8",
        missing_values="NA",
    )

    return data


@pytest.fixture(scope="module")
def be_vie_data_components(be_vie_data):
    """Convert the test data into a PModelEnv and arrays"""

    from pyrealm.pmodel import PModelEnvironment

    # Extract the key half hourly timestep variables
    ppfd_subdaily = be_vie_data["ppfd"]
    fapar_subdaily = be_vie_data["fapar"]
    datetime_subdaily = be_vie_data["time"].astype(np.datetime64)
    expected_gpp = be_vie_data["GPP_JAMES"]

    # Create the environment
    subdaily_env = PModelEnvironment(
        tc=be_vie_data["ta"],
        vpd=be_vie_data["vpd"],
        co2=be_vie_data["co2"],
        patm=be_vie_data["patm"],
    )

    return subdaily_env, ppfd_subdaily, fapar_subdaily, datetime_subdaily, expected_gpp


def test_FSPModel_JAMES(be_vie_data_components):
    """This tests the legacy calculations from the Mengoli et al JAMES paper, using that
    version of the weighted average calculations without acclimating xi."""

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
        expected_gpp[valid] * env.const.k_c_molmass,
        rtol=0.005,
    )


def test_FSPModel_corr(be_vie_data_components):
    """This tests the pyrealm implementation correlates well with the legacy
    calculations from the Mengoli et al JAMES paper without acclimating xi."""

    from pyrealm.pmodel import FastSlowPModel, FastSlowScaler

    env, ppfd, fapar, datetime, expected_gpp = be_vie_data_components

    # Get the fast slow scaler and set window
    fsscaler = FastSlowScaler(datetime)
    fsscaler.set_window(
        window_center=np.timedelta64(12, "h"),
        half_width=np.timedelta64(30, "m"),
    )

    # Fast slow model
    fs_pmodel = FastSlowPModel(
        env=env,
        fs_scaler=fsscaler,
        kphio=1 / 8,
        fapar=fapar,
        ppfd=ppfd,
    )

    valid = np.logical_not(
        np.logical_or(np.isnan(expected_gpp), np.isnan(fs_pmodel.gpp))
    )

    # Test that non-NaN predictions correlate well and are approximately the same
    gpp_in_micromols = fs_pmodel.gpp[valid] / env.const.k_c_molmass
    assert np.allclose(gpp_in_micromols, expected_gpp[valid], rtol=0.2)
    r_vals = np.corrcoef(gpp_in_micromols, expected_gpp[valid])
    assert np.alltrue(r_vals > 0.995)


@pytest.mark.parametrize("ndims", [2, 3, 4])
def test_FSPModel_dimensionality(be_vie_data, ndims):
    """This tests that the FastSlowPModel handles dimensions correctly.

    This broadcasts the BE-Vie onto more dimensions and checks that the code iterates
    over those dimensions correctly. fAPAR and PPFD are then fixed across the other
    dimensions to check the results scale as expected.
    """

    from pyrealm.pmodel import FastSlowPModel, FastSlowScaler, PModelEnvironment

    datetime = be_vie_data["time"].astype(np.datetime64)

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
    fs_pmodel = FastSlowPModel(
        env=env,
        fs_scaler=fsscaler,
        kphio=1 / 8,
        fapar=fapar_vals * np.ones(array_dims).transpose(),
        ppfd=np.ones(array_dims).transpose(),
    )

    # The GPP along the timescale of the different dimensions should be directly
    # proportional to the random fapar values and hence GPP/FAPAR should all equal the
    # value when it is set to 1
    timeaxis_mean = np.nansum(fs_pmodel.gpp, axis=0)
    assert np.allclose(timeaxis_mean / fapar_vals, timeaxis_mean[0])
