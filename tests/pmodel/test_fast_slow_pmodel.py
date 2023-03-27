from importlib import resources

import numpy as np
import pytest


@pytest.fixture
def be_vie_data():
    """Import the test data and provide it as a PModelEnv and two arrays"""

    from pyrealm.pmodel import PModelEnvironment

    with resources.path("data", "subdaily_BE_Vie_2014.csv") as data_path:
        data = np.genfromtxt(
            data_path,
            names=True,
            delimiter=",",
            dtype=None,
            encoding="UTF8",
            missing_values="NA",
        )

    # Extract the key half hourly timestep variables
    ppfd_subdaily = data["ppfd"]
    fapar_subdaily = data["fapar"]
    datetime_subdaily = data["time"].astype(np.datetime64)
    expected_gpp = data["GPP_JAMES"]

    # Create the environment
    subdaily_env = PModelEnvironment(
        tc=data["ta"],
        vpd=data["vpd"],
        co2=data["co2"],
        patm=data["patm"],
    )

    return subdaily_env, ppfd_subdaily, fapar_subdaily, datetime_subdaily, expected_gpp


def test_FSPModel_JAMES(be_vie_data):
    """This tests the legacy calculations from the Mengoli et al JAMES paper, using that
    version of the weighted average calculations without acclimating xi."""

    from pyrealm.pmodel import FastSlowScaler
    from pyrealm.pmodel.subdaily import FastSlowPModel_JAMES

    env, ppfd, fapar, datetime, expected_gpp = be_vie_data

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
    # and rounding of outputs prevent closer match
    assert np.allclose(fs_pmodel_james.gpp[valid], expected_gpp[valid], rtol=0.005)
