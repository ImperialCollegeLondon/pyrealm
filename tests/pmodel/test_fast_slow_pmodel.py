# import pandas
# import datetime
# import numpy as np
# from matplotlib import pyplot as plt
# from scipy.interpolate import interp1d
# import matplotlib.dates as mdates

# from pyrealm.pmodel import FastSlowScaler
# from pyrealm.pmodel import memory_effect, FastSlowPModel
# from pyrealm.pmodel import PModelEnvironment, PModel
# from pyrealm.pmodel.functions import (
#     calc_patm,
#     calc_ftemp_arrh,
#     calc_ftemp_kphio,
#     calc_viscosity_h2o,
# )
# from pyrealm.constants.pmodel_const import PModelConst


# fapar = pandas.read_csv(
#     "/Users/dorme/Research/REALM/P-model_subDaily-main/Data/data_Beni2_BE-Vie_2014.csv"
# )
# subdaily = pandas.read_csv(
#     "/Users/dorme/Research/REALM/P-model_subDaily-main/Data/FLX_BE-Vie_2014.csv",
#     na_values=[-9999],
# )

# # Get the times of the observations for the FluxNet data
# subdaily_datetime = [
#     datetime.datetime.strptime(str(x), "%Y%m%d%H%M")
#     for x in subdaily["TIMESTAMP_START"]
# ]
# subdaily_datetime = np.array(subdaily_datetime, dtype="datetime64[s]")

# # Extract the key half hourly timestep variables
# temp_subdaily = subdaily["TA_F"].to_numpy()
# vpd_subdaily = subdaily["VPD_F"].to_numpy() * 100  # Convert hPa to Pa
# co2_subdaily = subdaily["CO2_F_MDS"].to_numpy()
# ppfd_subdaily = subdaily["PPFD_IN"].to_numpy()

# # Convert the site elevation to a constant PATM
# patm_subdaily = np.ones_like(ppfd_subdaily) * calc_patm(493)

# fapar_dataset = pandas.read_csv(
#     "/Users/dorme/Research/REALM/P-model_subDaily-main/Data/data_Beni2_BE-Vie_2014.csv"
# )
# daily_fapar = fapar["fapar_spl"].to_numpy()

# # - interpolated between midnights, using np.nan for interpolation over last day
# fapar_subdaily_interpolator = interp1d(np.arange(365), daily_fapar, bounds_error=False)
# fapar_subdaily = fapar_subdaily_interpolator(np.arange(0, 365, 1 / 48))

# night = subdaily["NIGHT"].to_numpy().astype(np.bool_)

# fapar_subdaily[night] = 0
# ppfd_subdaily[night] = 0

# pmodel_subdaily_env = PModelEnvironment(
#     tc=temp_subdaily,
#     vpd=vpd_subdaily,
#     co2=co2_subdaily,
#     patm=patm_subdaily,
#     const=PModelConst(simple_viscosity=True),
# )

# fsscaler = FastSlowScaler(subdaily_datetime)

# # Sample acclimation conditions as the values within a one hour window centred on noon
# fsscaler.set_window(
#     window_center=np.timedelta64(12, "h"),
#     half_width=np.timedelta64(30, "m"),
# )

# fspmodel = FastSlowPModel(
#     env=pmodel_subdaily_env,
#     fs_scaler=fsscaler,
#     ppfd=ppfd_subdaily,
#     fapar=fapar_subdaily,
# )

from importlib import resources

import numpy as np
import pytest


@pytest.fixture
def be_vie_data():
    with resources.path("data", "subdaily_BE_Vie_2014.csv") as data_path:
        data = np.genfromtxt(
            data_path,
            names=True,
            delimiter=",",
            dtype=None,
            encoding="UTF8",
            missing_values="NA",
        )

    return data


def test_FSPModel_JAMES(be_vie_data):
    """This tests the legacy calculations from the Mengoli et al JAMES paper, using that
    version of the weighted average calculations without acclimating xi."""

    from pyrealm.pmodel import FastSlowScaler, PModelEnvironment
    from pyrealm.pmodel.subdaily import FastSlowPModel_JAMES

    # Extract the key half hourly timestep variables
    temp_subdaily = be_vie_data["ta"]
    vpd_subdaily = be_vie_data["vpd"]
    co2_subdaily = be_vie_data["co2"]
    patm_subdaily = be_vie_data["patm"]
    ppfd_subdaily = be_vie_data["ppfd"]
    fapar_subdaily = be_vie_data["fapar"]
    datetime_subdaily = be_vie_data["time"].astype(np.datetime64)
    expected_gpp = be_vie_data["GPP_JAMES"]

    # Create the environment
    subdaily_env = PModelEnvironment(
        tc=temp_subdaily,
        vpd=vpd_subdaily,
        co2=co2_subdaily,
        patm=patm_subdaily,
    )

    # Get the fast slow scaler and set window
    fsscaler = FastSlowScaler(datetime_subdaily)
    fsscaler.set_window(
        window_center=np.timedelta64(12, "h"),
        half_width=np.timedelta64(30, "m"),
    )

    # Alternate scalar used to duplicate VPD settings in JAMES implementation
    vpdscaler = FastSlowScaler(datetime_subdaily)
    vpdscaler.set_nearest(time=np.timedelta64(12, "h"))

    # Fast slow model without acclimating xi with best fit adaptations to the original
    # - VPD in daily optimum using different window
    # - Jmax and Vcmax filling from midday not window end
    fs_pmodel_james = FastSlowPModel_JAMES(
        env=subdaily_env,
        fs_scaler=fsscaler,
        kphio=1 / 8,
        fapar=fapar_subdaily,
        ppfd=ppfd_subdaily,
        vpd_scaler=vpdscaler,
        fill_from=np.timedelta64(12, "h"),
    )

    valid = np.logical_not(
        np.logical_or(np.isnan(expected_gpp), np.isnan(fs_pmodel_james.gpp))
    )

    # Test that non-NaN predictions are within 0.5% - slight differences in constants
    # and rounding of outputs prevent closer match
    assert np.allclose(fs_pmodel_james.gpp[valid], expected_gpp[valid], rtol=0.005)
