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


@pytest.fixture(scope="function")
def be_vie_data_components(be_vie_data):
    """Provides a data factory to convert the test data into a PModelEnv and arrays.

    This fixture returns an instance of a DataFactory class, that provides a `get`
    method to allow different subsets of the data to be built into the inputs for
    subdaily model testing. Providing this as a DataFactory generator allows tests to
    access more than one subset of the same data, which is useful in comparing the
    behaviour of complete and incomplete daily datetime sequences.

    The get method supports two modes, "pad" and "crop, both of which also require a
    `start` and `end` value.

    * In "pad" mode,  the original time series is extended by the specified number of
      half hourly steps at the start and end of the original data. The datetimes are
      filled in to give an actual time series but the data values are simply padded with
      `np.nan`. This is used to check that the presence of incomplete days does not
      affect the prediction of the sequence of GPP values. Since the actual data are not
      changed, the padded data should pass without affecting the calculations.

    * In "crop" mode, the original time series is cropped to only the rows in start:end.
      This is used to assess the behaviour of incomplete day handling and the switch
      points between providing daily estimates.
    """

    from pyrealm.pmodel import PModelEnvironment

    class DataFactory:
        def get(self, mode: str = "", start: int = 0, end: int = 0):
            # Get a copy of the data so as to not break the module scope loaded object.
            data = be_vie_data.copy()

            # Implement the two sampling modes
            if mode == "pad":
                # Get the new time series with the padded times
                datetime_subdaily = data["time"].to_numpy()
                spacing = np.diff(datetime_subdaily)[0]
                pad_start = datetime_subdaily[0] - np.arange(start, 0, -1) * spacing
                pad_end = datetime_subdaily[-1] + np.arange(1, end + 1, 1) * spacing

                # Pad the data frame with np.nan as requested
                data.index = range(start, len(data) + start)
                data = data.reindex(range(0, len(data) + start + end))

                # Set the new times into the data frame
                data["time"] = np.concatenate([pad_start, datetime_subdaily, pad_end])

            if mode == "crop":
                # Crop the data to the requested block and get the datetimes
                data = data.iloc[start:end]

            datetime_subdaily = data["time"].to_numpy()
            ppfd_subdaily = data["ppfd"].to_numpy()
            fapar_subdaily = data["fapar"].to_numpy()
            expected_gpp = data["GPP_JAMES"].to_numpy()

            # Create the environment including some randomly distributed water variables
            # to test the methods that require those variables
            rng = np.random.default_rng()
            subdaily_env = PModelEnvironment(
                tc=data["ta"].to_numpy(),
                vpd=data["vpd"].to_numpy(),
                co2=data["co2"].to_numpy(),
                patm=data["patm"].to_numpy(),
                theta=rng.uniform(low=0.5, high=0.8, size=ppfd_subdaily.shape),
                rootzonestress=rng.uniform(low=0.7, high=1.0, size=ppfd_subdaily.shape),
            )

            return (
                subdaily_env,
                ppfd_subdaily,
                fapar_subdaily,
                datetime_subdaily,
                expected_gpp,
            )

    return DataFactory()


def test_FSPModel_JAMES(be_vie_data_components):
    """Test FastSlowPModel_JAMES.

    This tests the legacy calculations from the Mengoli et al JAMES paper, using that
    version of the weighted average calculations without acclimating xi.
    """

    from pyrealm.pmodel import FastSlowScaler
    from pyrealm.pmodel.subdaily import FastSlowPModel_JAMES

    env, ppfd, fapar, datetime, expected_gpp = be_vie_data_components.get()

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
        allow_holdover=True,
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


@pytest.mark.parametrize(
    argnames="data_args",
    argvalues=[
        pytest.param({"mode": "pad", "start": 0, "end": 0}, id="no pad"),
        pytest.param({"mode": "pad", "start": 12, "end": 0}, id="start pad"),
        pytest.param({"mode": "pad", "start": 0, "end": 12}, id="end pad"),
        pytest.param({"mode": "pad", "start": 12, "end": 12}, id="pad both"),
    ],
)
def test_FSPModel_corr(be_vie_data_components, data_args):
    """Test FastSlowPModel.

    This tests that the pyrealm implementation including acclimating xi at least
    correlates well with the legacy calculations from the Mengoli et al JAMES paper
    without acclimating xi.

    The test also uses indirect parameterisation to pad the data to have incomplete days
    to check that the handling of the data holds up when the dates are changed. Note
    that this does nothing at all to the calculations - the data are padded with np.nan
    - so this is mostly checking that padding by itself does not cause the calculations
    to differ.
    """

    from pyrealm.pmodel import FastSlowScaler
    from pyrealm.pmodel.new_subdaily import SubdailyPModel

    env, ppfd, fapar, datetime, expected_gpp = be_vie_data_components.get(**data_args)

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
        allow_holdover=True,
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
        allow_holdover=True,
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

    env, ppfd, fapar, datetime, _ = be_vie_data_components.get()

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
        allow_holdover=True,
    )


@pytest.mark.parametrize("method_optchi", OPTIMAL_CHI_CLASS_REGISTRY.keys())
def test_convert_pmodel_to_subdaily(be_vie_data_components, method_optchi):
    """Tests the convert_pmodel_to_subdaily method."""

    from pyrealm.pmodel import FastSlowScaler, PModel
    from pyrealm.pmodel.new_subdaily import SubdailyPModel, convert_pmodel_to_subdaily

    env, ppfd, fapar, datetime, _ = be_vie_data_components.get()

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
        allow_holdover=True,
    )

    # Convert a standard model
    standard_model = PModel(env=env, kphio=1 / 8, method_optchi=method_optchi)
    standard_model.estimate_productivity(fapar=fapar, ppfd=ppfd)

    converted = convert_pmodel_to_subdaily(
        pmodel=standard_model, fs_scaler=fsscaler, allow_holdover=True
    )

    assert np.allclose(converted.gpp, direct.gpp, equal_nan=True)


@pytest.mark.parametrize(
    argnames="incomplete,complete,allow_partial",
    argvalues=[
        pytest.param(
            {"mode": "crop", "start": 48, "end": 48 * 6},
            {"mode": "crop", "start": 48, "end": 48 * 6},
            False,
            id="complete day",
        ),
        pytest.param(
            {"mode": "crop", "start": 36, "end": 48 * 6},
            {"mode": "crop", "start": 48, "end": 48 * 6},
            False,
            id="non window start",
        ),
        pytest.param(
            {"mode": "crop", "start": 25, "end": 48 * 6},
            {"mode": "crop", "start": 48, "end": 48 * 6},
            False,
            id="partial window start",
        ),
        pytest.param(
            {"mode": "crop", "start": 23, "end": 48 * 6},
            {"mode": "crop", "start": 0, "end": 48 * 6},
            False,
            id="all window start",
        ),
    ],
)
def test_FSPModel_incomplete_day_behaviour(
    be_vie_data_components,
    incomplete,
    complete,
    allow_partial,
):
    """Test FastSlowPModel.

    This tests that the SubdailyModel works as expected with incomplete start and end
    days and with partial data allowed or not. The setup fits two models, one of which
    uses complete days and the other of which

    """

    from pyrealm.pmodel.new_subdaily import FastSlowScaler, SubdailyPModel

    def model_fitter(env, ppfd, fapar, datetime):
        # Get the fast slow scaler and set window
        fsscaler = FastSlowScaler(datetime)
        fsscaler.set_window(
            window_center=np.timedelta64(12, "h"),
            half_width=np.timedelta64(30, "m"),
        )

        # Run as a subdaily model
        return SubdailyPModel(
            env=env,
            ppfd=ppfd,
            fapar=fapar,
            fs_scaler=fsscaler,
            allow_holdover=True,
        )

    # Feed the arguments for complete and incomplete days into DataFactory and then feed
    # the returned values (except the last element containing GPP) into the model fitter
    incomplete_mod = model_fitter(*be_vie_data_components.get(**incomplete)[:-1])
    complete_mod = model_fitter(*be_vie_data_components.get(**complete)[:-1])

    # Reduce the GPP values to those with matching timestamps
    incomplete_gpp = incomplete_mod.gpp[
        np.isin(incomplete_mod.datetimes, complete_mod.datetimes)
    ]
    complete_gpp = complete_mod.gpp[
        np.isin(complete_mod.datetimes, incomplete_mod.datetimes)
    ]

    assert np.allclose(incomplete_gpp, complete_gpp, equal_nan=True)