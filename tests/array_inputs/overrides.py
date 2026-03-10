"""Contains manual overrides in order to fix errors with the tests.

- `SKIP_METHODS` - a list of functions / methods to skip because they are not relevant
  or have issues that are difficult to resolve.

- `IGNORE_OUTPUTS` - a list of function results or class attributes to skip when
  checking for equality as they are not expected to be equal. This is only used in the
  broadcasting tests.

- `ADDITIONAL_INIT_METHODS` - a dictionary containing any additional methods that need
  to be used when initialising objects of that class.

- `REQUIRES` - a dictionary to define keyword arguments that must be used in each
  function / method. The values of the dictionary are functions that return the required
  keyword-argument pairs. This can also be used to avoid using default values for
  non-keyword arguments.

- `defined_method_args` - a function that returns manually defined arguments for
  functions / methods.
"""

from inspect import Parameter
from typing import Any

import numpy as np

from pyrealm.core.calendar import Calendar
from pyrealm.core.xarray import ArrayType
from pyrealm.demography.flora import PlantFunctionalType
from pyrealm.phenology.fapar_limitation import FaparLimitation, PhenologyConst
from tests.array_inputs.context import Context

# These methods are not relevant or are incompatible without additional work
SKIP_METHODS = [
    "evaluate_horner_polynomial",  # Coefficients are 1D
    # PModel
    "AcclimationModel.set_include",
    "PModel._get_daily_gpp",
    # Demography - mostly 1d arrays (dataframes)
    "CohortMethods.drop_cohort_data",
    "StemTraits",
    "StemTraits.drop_cohort_data",
    "_enforce_2D",
    "calculate_stem_projected_leaf_area_at_z",
    "get_crown_xy",
    "calculate_relative_crown_radius_at_z",
    "calculate_stem_projected_crown_area_at_z",
    "solve_canopy_area_filling_height",
    "calculate_crown_areas",
    "calculate_crown_fractions",
    "calculate_crown_r0",
    "calculate_crown_z_max",
    "calculate_dbh_from_height",
    "calculate_fine_root_respiration",
    "calculate_fine_root_turnover",
    "calculate_fine_root_masses",
    "calculate_foliage_masses",
    "calculate_foliage_turnover",
    "calculate_foliar_respiration",
    "calculate_gpp_topslice",
    "calculate_growth_increments",
    "calculate_heights",
    "calculate_net_primary_productivity",
    "calculate_reproductive_tissue_mass",
    "calculate_reproductive_tissue_respiration",
    "calculate_reproductive_tissue_turnover",
    "calculate_sapwood_masses",
    "calculate_sapwood_respiration",
    "calculate_stem_masses",
    "calculate_whole_crown_gpp",
]


# Ignore these outputs for broadcasting tests, they are not expected to be equal.
# Formats: [fn name] for function results, [class]:[attr] for class attributes
IGNORE_OUTPUTS = [
    "Cohorts:_cohort_id",
    "Calendar:n_dates",
]


# Call additional methods when initialising these classes
ADDITIONAL_INIT_METHODS = {
    "AcclimationModel": "set_nearest",
}


# The REQUIRES dictionary provides a way of populating required keywords arguments for
# some methods. The keys specify the object being created and the parent context in
# which it is created and the values provide a tuple of arguments that should be added
# to the method signature.


def _kwarg_params(names: tuple[str, ...]) -> dict[str, Parameter]:
    """Creates a dictionary containing inspect.Parameter instances."""
    return {
        name: Parameter(
            name=name,
            kind=Parameter.POSITIONAL_OR_KEYWORD,
            annotation=ArrayType,
        )
        for name in names
    }


REQUIRES: dict[tuple[str, tuple[str, ...]], dict[str, Parameter]] = {
    (
        "PModelEnvironment",
        (
            "OptimalChiC4RootzoneStress.estimate_chi",
            "OptimalChiC4RootzoneStress",
        ),
    ): _kwarg_params(("rootzonestress",)),
    (
        "PModelEnvironment",
        (
            "OptimalChiC4NoGammaRootzoneStress.estimate_chi",
            "OptimalChiC4NoGammaRootzoneStress",
        ),
    ): _kwarg_params(("rootzonestress",)),
    (
        "PModelEnvironment",
        (
            "OptimalChiPrentice14RootzoneStress.estimate_chi",
            "OptimalChiPrentice14RootzoneStress",
        ),
    ): _kwarg_params(("rootzonestress",)),
    (
        "PModelEnvironment",
        (
            "OptimalChiLavergne20C3.estimate_chi",
            "OptimalChiLavergne20C3",
        ),
    ): _kwarg_params(("theta",)),
    (
        "PModelEnvironment",
        (
            "OptimalChiLavergne20C4.estimate_chi",
            "OptimalChiLavergne20C4",
        ),
    ): _kwarg_params(("theta",)),
    (
        "PModelEnvironment",
        ("QuantumYieldSandoval", "QuantumYieldSandoval"),
    ): _kwarg_params(("aridity_index", "mean_growth_temperature")),
    (
        "PModelEnvironment",
        (
            "QuantumYieldSandoval.peak_quantum_yield",
            "QuantumYieldSandoval",
        ),
    ): _kwarg_params(("aridity_index", "mean_growth_temperature")),
    # Don't use default xi_values (None) in estimate_chi
    ("OptimalChiPrentice14.estimate_chi", ()): _kwarg_params(("xi_values",)),
    ("OptimalChiPrentice14RootzoneStress.estimate_chi", ()): _kwarg_params(
        ("xi_values",)
    ),
    ("OptimalChiC4.estimate_chi", ()): _kwarg_params(("xi_values",)),
    ("OptimalChiC4RootzoneStress.estimate_chi", ()): _kwarg_params(("xi_values",)),
    ("OptimalChiLavergne20C3.estimate_chi", ()): _kwarg_params(("xi_values",)),
    ("OptimalChiLavergne20C4.estimate_chi", ()): _kwarg_params(("xi_values",)),
    ("OptimalChiC4NoGamma.estimate_chi", ()): _kwarg_params(("xi_values",)),
    ("OptimalChiC4NoGammaRootzoneStress.estimate_chi", ()): _kwarg_params(
        ("xi_values",)
    ),
}


def _get_time_dim(ctx: Context) -> int | None:
    """Get the time dimension for the current array (first dimension of full shape)."""
    time_dim: int | None = 0
    if ctx.array_type == "xarray":
        time_dim_key = "a"
        # Determine if and where it is in this argument
        if time_dim_key in ctx.array_dims:
            time_dim = ctx.array_dims.index(time_dim_key)
        else:
            time_dim = None
    return time_dim


# These methods require specific arguments
def defined_method_args(argument: str, ctx: Context) -> Any | None:
    """Return any manually defined arguments for the current function / method.

    This is done by defining an `arguments` dictionary for the function and then
    returning the value (if any) of the specific `argument`.

    Args:
        argument (str): The name of the input argument to define.
        ctx (Context): The context containing the name of the function (`ctx.name`) and
        the parent classes / class method being tested (`ctx.parents`).

    Returns:
        Any | None: The manually defined value for the argument, or `None` if it can be
        set by the defaults.
    """
    shape = ctx.shape
    bcast_shape = ctx.bcast_shape

    # PModel parameters
    splashDatesLen = 10
    # Demography parameters
    n_pft = 3
    n_heights = 2
    pft_names = [f"Tree{i + 1}" for i in range(n_pft)]

    fapar_limitation = FaparLimitation(
        annual_total_potential_gpp=np.ones(48),
        annual_mean_ca=np.ones(48),
        annual_mean_chi=np.ones(48),
        annual_mean_vpd=np.ones(48),
        annual_total_precip=np.ones(48),
        annual_growing_season_length=np.ones(48),
        aridity_index=np.ones(48),
        years=np.zeros((48,), dtype="datetime64[Y]"),
        phenology_const=PhenologyConst(
            z=12.227, k=0.5, f0_coefficients=(0.65, 0.604169, 1.9), sigma=0.771
        ),
    )

    method_arguments_list: dict[str, dict] = {
        "broadcast_time": {"shape": bcast_shape},
        ## PModel
        # Subdaily data needs more than 1 day of times (uses 48 hours)
        "AcclimationModel": {"datetimes": np.arange(0, 48, dtype="datetime64[h]")},
        "AcclimationModel.set_nearest": {"time": np.timedelta64(12, "h")},
        "AcclimationModel._validate_and_set_datetimes": {
            "datetimes": np.arange(0, 48, dtype="datetime64[h]")
        },
        "AcclimationModel._get_subdaily_interpolation_xy": {"values": np.ones(2)},
        "AcclimationModel.fill_daily_to_subdaily": {"values": np.ones((2, *shape[1:]))},
        "AcclimationModel.get_window_values": {"values": np.ones(48)},
        "AcclimationModel.get_daily_means": {"values": np.ones(48)},
        "calculate_kattge_knorr_arrhenius_factor": {
            "coef": {"ha": 1, "hd": 1, "entropy_intercept": 1, "entropy_slope": 1}
        },
        "SplashModel.estimate_daily_water_balance": {
            "previous_wn": np.full((splashDatesLen, *shape[1:]), 10),
        },
        "SplashModel.calculate_soil_moisture": {"wn_init": np.full(shape[1:], 10)},
        "PModelEnvironment": {"patm": np.full(shape, 100000)},
        "TwoLeafIrradiance": {"patm": np.full(shape, 100000)},
        ## Demography uses 1D arrays (a lot of these could probably be skipped)
        "Cohorts": {
            "dbh_values": np.full(n_pft, 2),
            "n_individuals": np.ones(n_pft),
            "pft_names": np.array(pft_names, dtype=np.str_),
        },
        "Flora": {"pfts": [PlantFunctionalType(name=name) for name in pft_names]},
        "Flora.get_stem_traits": {"pft_names": pft_names},
        "Canopy": {"fit_ppa": True},
        "CohortCanopyData": {
            "projected_leaf_area": np.ones((n_heights, n_pft)),
            "n_individuals": np.ones(n_pft),
            "lai": np.ones(n_pft),
            "par_ext": np.ones(n_pft),
        },
        "CommunityCanopyData": {
            "absorption": np.full((n_heights, n_pft), 0.5),
            "leaf_area_index": np.full((n_heights, n_pft), 0.5),
            "cohort_leaf_area": np.full((n_heights, n_pft), 1),
        },
        "StemAllometry": {"at_dbh": np.full(n_pft, 0.5)},
        "StemAllocation": {"whole_crown_gpp": np.full(n_pft, 0.5)},
        "CrownProfile": {"z": np.linspace(5, 15, n_heights)[:, np.newaxis]},
        "Cohorts.drop_cohort_data": {"drop_indices": [0, 1]},
        "StemAllometry.drop_cohort_data": {"drop_indices": [0, 1]},
        "Community.drop_cohorts": {"drop_indices": [0, 1]},
        ## Phenology
        "FaparLimitation": {"years": np.ones(bcast_shape[0], dtype="datetime64[Y]")},
        "Phenology": {
            "daily_gpp": np.full((48,), 0.5),
            "datetimes": np.arange(0, 48, dtype="datetime64[D]"),
            "fapar_limitation": fapar_limitation,
        },
    }
    arguments: dict = method_arguments_list.get(ctx.name, {})

    # Arguments that use temporary variables or depend on parents

    if ctx.name.split(".")[0] == "AnnualValueCalculator":
        # The shapes of many of the inputs are required to match `data_shape`
        nTime = 3
        data_shape = (nTime, *bcast_shape[1:])
        if ctx.name == "AnnualValueCalculator":
            arguments = {
                "data_shape": data_shape,
                "timing": np.arange(0, nTime, dtype="datetime64[D]"),
            }
        elif ctx.name in [
            "AnnualValueCalculator._split_values_by_year",
            "AnnualValueCalculator.get_annual_means",
            "AnnualValueCalculator.get_annual_totals",
        ]:
            arguments = {"values": np.ones(data_shape)}

    if ctx.name.split(".")[0] in ["DailySolarFluxes", "DailyEvapFluxes"]:
        # Needs first dimension to match the dates
        nTime = 4
        shape2 = (1 if shape[0] == 1 else nTime, *shape[1:])
        if ctx.name == "DailySolarFluxes":
            arguments = {
                "dates": Calendar(np.arange(0, nTime, dtype="datetime64[D]")),
                "latitude": np.full(shape2, 10),
                "elevation": np.full(shape2, 10),
                "sunshine_fraction": np.full(shape2, 0.5),
                "temperature": np.full(shape2, 25),
            }
        elif ctx.name == "DailyEvapFluxes":
            arguments = {
                "pa": np.full(shape2, 10),
                "tc": np.full(shape2, 25),
                "kWm": np.full(shape2, 150),
            }
        elif ctx.name == "DailyEvapFluxes.estimate_aet":
            arguments = {"wn": np.full(shape2, 10)}

    if ctx.name == "SplashModel":
        if (
            ctx.parents
            and ctx.parents[0] == "SplashModel.estimate_initial_soil_moisture"
        ):
            # Requires at least 1 year of data
            nTime = 366
        else:
            nTime = splashDatesLen
        splashShape = (1 if shape[0] == 1 else nTime, *shape[1:])
        arguments = {
            "dates": Calendar(np.arange(0, nTime, dtype="datetime64[D]")),
            "lat": np.full(splashShape, 10),
            "elv": np.full(splashShape, 10),
            "sf": np.full(splashShape, 0.5),
            "tc": np.full(splashShape, 25),
            "pn": np.full(splashShape, 10),
        }

    if ctx.name == "PModelEnvironment":
        if ctx.parents and ctx.parents[-1] == "SubdailyPModel":
            # SubdailyPModel needs more than 1 day (uses 48 hourly times)

            # Replace the time dimension (the first dimension)
            envShape = list(shape)
            time_dim = _get_time_dim(ctx)
            if time_dim is not None:
                envShape[time_dim] = 1 if shape[time_dim] == 1 else 48

            arguments = {
                "tc": np.full(envShape, 20),
                "vpd": np.full(envShape, 40),
                "co2": np.full(envShape, 1000),
                "patm": np.full(envShape, 101325),
                "fapar": np.full(envShape, 1),
                "ppfd": np.full(envShape, 800),
            }

    if ctx.name == "SubdailyPModel.apply_gpp_penalty_factor":
        # SubdailyPModel needs more than 1 day (uses 48 hourly times)
        _shape = list(shape)
        time_dim = _get_time_dim(ctx)
        if time_dim is not None:
            _shape[time_dim] = 1 if shape[time_dim] == 1 else 48
        arguments = {"penalty_factor": np.full(_shape, 0.5)}

    return arguments.get(argument)
