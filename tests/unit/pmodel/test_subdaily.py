"""Tests the implementation of the FastSlowModel against the reference benchmark."""

from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyrealm.pmodel.optimal_chi import OPTIMAL_CHI_CLASS_REGISTRY


@pytest.mark.parametrize(
    argnames="data_args",
    argvalues=[
        pytest.param({"mode": "pad", "start": 0, "end": 0}, id="no pad"),
        pytest.param({"mode": "pad", "start": 12, "end": 0}, id="start pad"),
        pytest.param({"mode": "pad", "start": 0, "end": 12}, id="end pad"),
        pytest.param({"mode": "pad", "start": 12, "end": 12}, id="pad both"),
    ],
)
def test_SubdailyPModel_corr(be_vie_data_components, data_args):
    """Test SubdailyPModel.

    This tests that the pyrealm implementation including acclimating xi at least
    correlates well with the legacy calculations from the Mengoli et al JAMES paper
    without acclimating xi.

    The test also uses indirect parameterisation to pad the data to have incomplete days
    to check that the handling of the data holds up when the dates are changed. Note
    that this does nothing at all to the calculations - the data are padded with np.nan
    - so this is mostly checking that padding by itself does not cause the calculations
    to differ.
    """

    from pyrealm.pmodel.acclimation import AcclimationModel
    from pyrealm.pmodel.pmodel import SubdailyPModel

    env, datetime, expected_gpp = be_vie_data_components.get(**data_args)

    # Get the acclimation model and set window
    acclim_model = AcclimationModel(datetime, allow_holdover=True)
    acclim_model.set_window(
        window_center=np.timedelta64(12, "h"),
        half_width=np.timedelta64(30, "m"),
    )

    # Run as a subdaily model using the kphio used in the reference implementation.
    subdaily_pmodel = SubdailyPModel(
        env=env,
        reference_kphio=1 / 8,
        acclim_model=acclim_model,
    )

    valid = np.logical_not(
        np.logical_or(np.isnan(expected_gpp), np.isnan(subdaily_pmodel.gpp))
    )

    # Test that non-NaN predictions correlate well and are approximately the same
    gpp_in_micromols = subdaily_pmodel.gpp[valid] / env.core_const.k_c_molmass
    assert_allclose(gpp_in_micromols, expected_gpp[valid], rtol=0.2)
    r_vals = np.corrcoef(gpp_in_micromols, expected_gpp[valid])
    assert np.all(r_vals > 0.995)


@pytest.mark.parametrize(
    argnames="previous_realised, outcome, msg",
    argvalues=[
        pytest.param(
            {"xi": np.ones(1), "jmax25": np.ones(1), "vcmax25": np.ones(1)},
            does_not_raise(),
            None,
            id="good",
        ),
        pytest.param(
            (np.ones(1), np.ones(1), np.ones(1)),
            pytest.raises(ValueError),
            "previous_realised must be a dictionary of arrays, with entries "
            "for 'xi', 'jmax25' and 'vcmax25'.",
            id="not a dict",
        ),
        pytest.param(
            {"xi": np.ones(1), "j_max25": np.ones(1), "v_cmax25": np.ones(1)},
            pytest.raises(ValueError),
            "previous_realised must be a dictionary of arrays, with entries "
            "for 'xi', 'jmax25' and 'vcmax25'.",
            id="dict with wrong keys",
        ),
        pytest.param(
            {"xi": 1, "j_max25": 1, "v_cmax25": 1},
            pytest.raises(ValueError),
            "previous_realised must be a dictionary of arrays, with entries "
            "for 'xi', 'jmax25' and 'vcmax25'.",
            id="dict with non array values",
        ),
        pytest.param(
            {"xi": np.ones(2), "jmax25": np.ones(2), "vcmax25": np.ones(2)},
            pytest.raises(ValueError),
            "`previous_realised` arrays have wrong shape in SubdailyPModel",
            id="dict with badly sized arrays",
        ),
    ],
)
def test_SubdailyPModel_previous_realised_validation(
    be_vie_data_components, previous_realised, outcome, msg
):
    """Test the functionality that allows the subdaily model to restart in blocks."""

    from pyrealm.pmodel.acclimation import AcclimationModel
    from pyrealm.pmodel.pmodel import SubdailyPModel

    env, datetime, expected_gpp = be_vie_data_components.get()

    # Get the acclimation model and set window
    acclim_model = AcclimationModel(datetime, allow_holdover=True)
    acclim_model.set_window(
        window_center=np.timedelta64(12, "h"),
        half_width=np.timedelta64(30, "m"),
    )

    # Run as a subdaily model
    with outcome as excep:
        _ = SubdailyPModel(
            env=env,
            acclim_model=acclim_model,
            previous_realised=previous_realised,
        )

    if not isinstance(outcome, does_not_raise):
        assert str(excep.value) == msg


def test_SubdailyPModel_previous_realised(be_vie_data_components):
    """Test the functionality that allows the subdaily model to restart in blocks."""

    from pyrealm.pmodel.acclimation import AcclimationModel
    from pyrealm.pmodel.pmodel import SubdailyPModel

    # Run all in one model
    env, datetime, _ = be_vie_data_components.get(mode="crop", start=0, end=17520)

    # Get the acclimation model and set window
    acclim_model = AcclimationModel(datetime, allow_holdover=True)
    acclim_model.set_window(
        window_center=np.timedelta64(12, "h"),
        half_width=np.timedelta64(30, "m"),
    )

    # Run as a subdaily model using the kphio used in the reference implementation.
    all_in_one_subdaily_pmodel = SubdailyPModel(
        env=env,
        reference_kphio=1 / 8,
        acclim_model=acclim_model,
    )

    # Run first half of year
    env1, datetime1, _ = be_vie_data_components.get(mode="crop", start=0, end=182 * 48)

    # Get the acclimation model
    acclim_model1 = AcclimationModel(datetime1, allow_holdover=True)
    acclim_model1.set_window(
        window_center=np.timedelta64(12, "h"),
        half_width=np.timedelta64(30, "m"),
    )

    # Run as a subdaily model using the kphio used in the reference implementation.
    part_1_subdaily_pmodel = SubdailyPModel(
        env=env1,
        reference_kphio=1 / 8,
        acclim_model=acclim_model1,
    )

    # Run second year
    env2, datetime2, _ = be_vie_data_components.get(
        mode="crop", start=182 * 48, end=365 * 48
    )

    # Get the acclimation model
    acclim_model2 = AcclimationModel(datetime2, allow_holdover=True)
    acclim_model2.set_window(
        window_center=np.timedelta64(12, "h"),
        half_width=np.timedelta64(30, "m"),
    )

    # Run as a subdaily model using the kphio used in the reference implementation.
    # Note the explicit use of [[-1]] to give array values, not scalar np.float
    part_2_subdaily_pmodel = SubdailyPModel(
        env=env2,
        reference_kphio=1 / 8,
        acclim_model=acclim_model2,
        previous_realised={
            "xi": part_1_subdaily_pmodel.xi_daily_realised[[-1]],
            "vcmax25": part_1_subdaily_pmodel.vcmax25_daily_realised[[-1]],
            "jmax25": part_1_subdaily_pmodel.jmax25_daily_realised[[-1]],
        },
    )

    assert_allclose(
        all_in_one_subdaily_pmodel.gpp,
        np.concat([part_1_subdaily_pmodel.gpp, part_2_subdaily_pmodel.gpp]),
        equal_nan=True,
    )


@pytest.mark.parametrize("ndims", [2, 3, 4])
def test_SubdailyPModel_dimensionality(be_vie_data, ndims):
    """Tests that the SubdailyPModel handles dimensions correctly.

    This broadcasts the BE-Vie onto more dimensions and checks that the code iterates
    over those dimensions correctly. fAPAR and PPFD are then fixed across the other
    dimensions to check the results scale as expected.
    """

    from pyrealm.pmodel import PModelEnvironment
    from pyrealm.pmodel.acclimation import AcclimationModel
    from pyrealm.pmodel.pmodel import SubdailyPModel

    datetime = be_vie_data["time"].to_numpy()

    # Set up the dimensionality for the test - create a shape tuple with extra
    # dimensions and then broadcast the model inputs onto it. These need to be
    # transposed to return datetime to the first axis. When n_dim = 1, the data are
    # passed as is.
    extra_dims = [3] * (ndims - 1)
    array_dims = tuple([*extra_dims, len(datetime)])

    # Apply a different random value of fAPAR for each time series, but set a single
    # element to 1.0 as a reference value
    fapar_vals = np.random.random(extra_dims)
    fapar_vals.flat[0] = 1.0

    # Create the environment
    env = PModelEnvironment(
        tc=np.broadcast_to(be_vie_data["ta"], array_dims).transpose(),
        vpd=np.broadcast_to(be_vie_data["vpd"], array_dims).transpose(),
        co2=np.broadcast_to(be_vie_data["co2"], array_dims).transpose(),
        patm=np.broadcast_to(be_vie_data["patm"], array_dims).transpose(),
        fapar=fapar_vals * np.ones(array_dims).transpose(),
        ppfd=np.ones(array_dims).transpose(),
    )

    # Get the acclimation model and set window
    acclim_model = AcclimationModel(datetime, allow_holdover=True)
    acclim_model.set_window(
        window_center=np.timedelta64(12, "h"),
        half_width=np.timedelta64(30, "m"),
    )

    # Fast slow model
    subdaily_pmodel = SubdailyPModel(
        env=env,
        acclim_model=acclim_model,
    )

    # The GPP along the timescale of the different dimensions should be directly
    # proportional to the random fapar values and hence GPP/FAPAR should all equal the
    # value when it is set to 1
    timeaxis_mean = np.nansum(subdaily_pmodel.gpp, axis=0)
    assert_allclose(
        timeaxis_mean / fapar_vals, np.full_like(timeaxis_mean, timeaxis_mean.flat[0])
    )


@pytest.mark.parametrize("method_optchi", OPTIMAL_CHI_CLASS_REGISTRY.keys())
def test_Subdaily_opt_chi_methods(be_vie_data_components, method_optchi):
    """Tests that the SubdailyModel runs with all the provided optimal chi classes.

    This currently just checks that the subdaily model _runs_ with each of the different
    implementations of the OptimalChi ABC.
    """

    from pyrealm.pmodel.acclimation import AcclimationModel
    from pyrealm.pmodel.pmodel import SubdailyPModel

    env, datetime, _ = be_vie_data_components.get()

    # Get the acclimation model and set window
    acclim_model = AcclimationModel(datetime, allow_holdover=True)
    acclim_model.set_window(
        window_center=np.timedelta64(12, "h"),
        half_width=np.timedelta64(30, "m"),
    )

    # Run as a subdaily model and it should complete.
    _ = SubdailyPModel(
        env=env,
        acclim_model=acclim_model,
        method_optchi=method_optchi,
    )


@pytest.mark.parametrize("method_optchi", OPTIMAL_CHI_CLASS_REGISTRY.keys())
def test_convert_pmodel_to_subdaily(be_vie_data_components, method_optchi):
    """Tests the convert_pmodel_to_subdaily method."""

    from pyrealm.pmodel.acclimation import AcclimationModel
    from pyrealm.pmodel.pmodel import PModel, SubdailyPModel

    env, datetime, _ = be_vie_data_components.get()

    # Get the acclimation model and set window
    acclim_model = AcclimationModel(datetime, allow_holdover=True)
    acclim_model.set_window(
        window_center=np.timedelta64(12, "h"),
        half_width=np.timedelta64(30, "m"),
    )

    # Run as a subdaily model
    direct = SubdailyPModel(
        env=env,
        acclim_model=acclim_model,
        method_optchi=method_optchi,
    )

    # Convert a standard model
    standard_model = PModel(
        env=env,
        method_optchi=method_optchi,
    )

    converted = standard_model.to_subdaily(acclim_model=acclim_model)

    assert_allclose(converted.gpp, direct.gpp, equal_nan=True)


@pytest.mark.parametrize(
    argnames="incomplete,complete,allow_holdover,allow_partial_data,patch_means,raises",
    argvalues=[
        pytest.param(
            {"mode": "crop", "start": 48, "end": 48 * 6},
            {"mode": "crop", "start": 48, "end": 48 * 6},
            True,
            False,
            False,
            does_not_raise(),
            id="complete day hold+ partial-",
        ),
        pytest.param(
            {"mode": "crop", "start": 36, "end": 48 * 6},
            {"mode": "crop", "start": 48, "end": 48 * 6},
            True,
            False,
            False,
            does_not_raise(),
            id="non window start hold+ partial-",
        ),
        pytest.param(
            {"mode": "crop", "start": 25, "end": 48 * 6},
            {"mode": "crop", "start": 48, "end": 48 * 6},
            True,
            False,
            False,
            does_not_raise(),
            id="partial window start hold+ partial-",
        ),
        pytest.param(
            {"mode": "crop", "start": 23, "end": 48 * 6},
            {"mode": "crop", "start": 0, "end": 48 * 6},
            True,
            False,
            False,
            does_not_raise(),
            id="all window start hold+ partial-",
        ),
        pytest.param(
            {"mode": "crop", "start": 48, "end": 48 * 6},
            {"mode": "crop", "start": 48, "end": 48 * 6},
            False,
            False,
            False,
            does_not_raise(),
            id="complete day hold- partial-",
        ),
        pytest.param(
            {"mode": "crop", "start": 36, "end": 48 * 6},
            {"mode": "crop", "start": 48, "end": 48 * 6},
            False,
            False,
            False,
            pytest.raises(ValueError),
            id="non window start hold- partial-",
        ),
        pytest.param(
            {"mode": "crop", "start": 25, "end": 48 * 6},
            {"mode": "crop", "start": 48, "end": 48 * 6},
            False,
            False,
            False,
            pytest.raises(ValueError),
            id="partial window start hold- partial-",
        ),
        pytest.param(
            {"mode": "crop", "start": 23, "end": 48 * 6},
            {"mode": "crop", "start": 0, "end": 48 * 6},
            False,
            False,
            False,
            does_not_raise(),
            id="all window start hold- partial-",
        ),
        pytest.param(
            {"mode": "crop", "start": 48, "end": 48 * 6},
            {"mode": "crop", "start": 48, "end": 48 * 6},
            True,
            True,
            False,
            does_not_raise(),
            id="complete day hold+ partial+",
        ),
        pytest.param(
            {"mode": "crop", "start": 36, "end": 48 * 6},
            {"mode": "crop", "start": 48, "end": 48 * 6},
            True,
            True,
            False,
            does_not_raise(),
            id="non window start hold+ partial+",
        ),
        pytest.param(
            {"mode": "crop", "start": 25, "end": 48 * 6},
            {"mode": "crop", "start": 0, "end": 48 * 6},
            True,
            True,
            True,
            does_not_raise(),
            id="partial window start hold+ partial+",
        ),
        pytest.param(
            {"mode": "crop", "start": 23, "end": 48 * 6},
            {"mode": "crop", "start": 0, "end": 48 * 6},
            True,
            True,
            False,
            does_not_raise(),
            id="all window start hold+ partial+",
        ),
    ],
)
def test_SubdailyPModel_incomplete_day_behaviour(
    mocker,
    be_vie_data_components,
    incomplete,
    complete,
    allow_holdover,
    allow_partial_data,
    patch_means,
    raises,
):
    """Test SubdailyPModel.

    This tests that the SubdailyModel works as expected with incomplete start and end
    days and with partial data and holdover allowed or not. The setup fits two models:

    * complete_mod uses only complete days, as in the original implementation
    * incomplete days can be:
        * actually complete (should be identical)
        * include only non-window observations at the end of the first day
        * include partial window observations on the first day
        * include all of the window observations on the first day

    To then explain the test logic, the first check is if the incomplete model raises
    or not:

    * With allow_partial_data = True and allow_holdover = True, all models should run,
      but with allow_partial_data = False and allow_holdover = False, the non-window and
      partial models will fail because they will include an np.nan estimate for the
      first incomplete day.

    * With allow_partial_data = True and allow_holdover = True, all models should run,
      but with allow_partial_data = True and allow_holdover = False, the non-window
      models will fail because it includes an np.nan estimate for the first incomplete
      day.

    The next check is whether the GPP predictions then match to a suitably chosen
    complete day:

    * For the complete 'incomplete' this should just match.
    * For the non-window incomplete, it starts predictions the following day, so a
      complete model starting the following midnight should match. The same is true for
      the partial window if allow_partial_data = False - it's basically the same model.
    * The all window model should match a complete model starting at the previous
      midnight.
    * It is more difficult to test the partial data with allow_partial_data = True
      because the subsample of available data gives a different daily mean to the
      complete model. You can force all the values to be equal to the mean within the
      window - the mean is now stable - but then the half hourly predictions are wrong.
      The only practical way to do this is to patch `get_daily_means` with the daily
      averages for each variable in the complete model, but allow the rest of the
      calculations to use the actual data.
    """

    from pyrealm.pmodel.acclimation import AcclimationModel
    from pyrealm.pmodel.pmodel import SubdailyPModel

    def model_fitter(env, datetime):
        # Get the acclimation model and set window
        acclim_model = AcclimationModel(
            datetime,
            allow_holdover=allow_holdover,
            allow_partial_data=allow_partial_data,
        )
        acclim_model.set_window(
            window_center=np.timedelta64(12, "h"),
            half_width=np.timedelta64(30, "m"),
        )

        # Run as a subdaily model
        return SubdailyPModel(env=env, acclim_model=acclim_model)

    # Feed the arguments for complete and incomplete days into DataFactory and then feed
    # the returned values (except the last element containing GPP) into the model fitter
    # function above
    with raises:
        complete_mod = model_fitter(*be_vie_data_components.get(**complete)[:-1])

        if patch_means:
            # Patch the return values for `get_daily_means` to be the same as the
            # complete model - these have to to be in the same order that they are
            # calculated within the model, so that each call gets patched with the right
            # values.
            patched_means = [
                complete_mod.pmodel_acclim.env.tc,
                complete_mod.pmodel_acclim.env.co2,
                complete_mod.pmodel_acclim.env.patm,
                complete_mod.pmodel_acclim.env.vpd,
                complete_mod.pmodel_acclim.env.fapar,
                complete_mod.pmodel_acclim.env.ppfd,
            ]

            with mocker.patch.object(
                AcclimationModel,
                "get_daily_means",
                side_effect=patched_means,
            ):
                incomplete_mod = model_fitter(
                    *be_vie_data_components.get(**incomplete)[:-1]
                )
        else:
            incomplete_mod = model_fitter(
                *be_vie_data_components.get(**incomplete)[:-1]
            )

    if isinstance(raises, does_not_raise):
        # Reduce the GPP values to those with matching timestamps
        incomplete_gpp = incomplete_mod.gpp[
            np.isin(
                incomplete_mod.acclim_model.datetimes,
                complete_mod.acclim_model.datetimes,
            )
        ]
        complete_gpp = complete_mod.gpp[
            np.isin(
                complete_mod.acclim_model.datetimes,
                incomplete_mod.acclim_model.datetimes,
            )
        ]

        # Check the predictions are close
        assert_allclose(incomplete_gpp, complete_gpp, equal_nan=True)
