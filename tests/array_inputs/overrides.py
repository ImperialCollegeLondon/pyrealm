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

- `MANUAL_ARGS` - a dictionary containing manually defined arguments for functions /
  methods where the automatic generation doesn't work. This is used to resolve most of
  the errors.
"""

from collections.abc import Callable
from inspect import Parameter
from math import ceil

import numpy as np

from pyrealm.constants import PhenologyConst
from pyrealm.core.calendar import Calendar
from pyrealm.core.xarray import ArrayType
from pyrealm.demography.flora import PlantFunctionalType
from pyrealm.phenology.fapar_limitation import FaparLimitation
from tests.array_inputs.context import Context

# These methods are not relevant or are incompatible without additional work
SKIP_METHODS = [
    "evaluate_horner_polynomial",  # Coefficients are 1D
    # PModel
    "AcclimationModel.set_include",
    "PModel._get_daily_gpp",
    # OK - this is really problematic. Array auto-discovery gets hung up on the
    # internals of pydantic - not sure the greedy approach to what gets tested is
    # sustainable.
    "Flora.__repr_args__",
    # Something about pandas methods on Cohort objects also triggers - these are
    # explicitly blocked in utils.get_method_list rather than handling each one here.
    "create_cohorts",
    "Cohorts",
    "Cohorts.drop_cohort_data",
    "CrownProfile",
    "CrownProfile.to_xy",
    "Canopy",
    # Demography - mostly 1d arrays (dataframes)
    "StemAllocation",
    "StemAllometry",
    "StemMaintenance",
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


# Everything below here populates the MANUAL_ARGS dictionary.
# This is used to define manual arguments for functions / methods where the automatic
# argument generation fails.

# The dictionary values are functions that take the Context and return a parameter dict.
# Use the register_args decorator to populate this.
MANUAL_ARGS: dict[str, Callable[[Context], dict]] = {}


def register_args(func_name: str | list[str]):
    """Decorator for adding functions to MANUAL_ARGS.

    This can take a list of function names if they require the same manual arguments.
    This will also check to make sure the same function isn't accidentally added twice.
    """

    def check_empty(func_name: str):
        if func_name in MANUAL_ARGS:
            msg = f"Multiple entries for {func_name} in MANUAL_ARGS."
            raise RuntimeError(msg)

    def decorator(args_dict):
        if isinstance(func_name, str):
            check_empty(func_name)
            MANUAL_ARGS[func_name] = args_dict
        elif isinstance(func_name, list):
            for name in func_name:
                check_empty(name)
                MANUAL_ARGS[name] = args_dict

    return decorator


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


def _set_time_len(n_time: int, ctx: Context, allow_one: bool = True) -> tuple[int, ...]:
    """Set the time dimension (first dimension of full shape) to either n_time or 1."""
    shape = list(ctx.shape)
    time_dim = _get_time_dim(ctx)
    if time_dim is not None:
        shape[time_dim] = 1 if (ctx.shape[time_dim] == 1 and allow_one) else n_time
    return tuple(shape)


def _del_time_dim(ctx: Context) -> tuple[int, ...]:
    """Return the shape with the time dimension removed."""
    shape = list(ctx.shape)
    time_dim = _get_time_dim(ctx)
    if time_dim is not None:
        del shape[time_dim]
    return tuple(shape)


## Core module

register_args("broadcast_time")(lambda ctx: {"shape": ctx.bcast_shape})


## PModel module
_PATM = 101325  # The automatic value (1) gives an error
_SUBDAILY_N_TIMES = 48  # More than a day needed and needs to match across methods

register_args("TwoLeafIrradiance")(lambda ctx: {"patm": np.full(ctx.shape, _PATM)})


@register_args("PModelEnvironment")
def _(ctx):
    if ctx.parents and ctx.parents[-1] == "SubdailyPModel":
        shape = _set_time_len(_SUBDAILY_N_TIMES, ctx)
        return {
            "tc": np.full(shape, 20),
            "vpd": np.full(shape, 40),
            "co2": np.full(shape, 1000),
            "patm": np.full(shape, _PATM),
            "fapar": np.full(shape, 1),
            "ppfd": np.full(shape, 800),
        }

    else:
        return {"patm": np.full(ctx.shape, _PATM)}


register_args("SubdailyPModel.apply_gpp_penalty_factor")(
    lambda ctx: {"penalty_factor": np.full(_set_time_len(_SUBDAILY_N_TIMES, ctx), 0.5)}
)


# AnnualValueCalculator
# The shapes of many inputs are required to match the `data_shape` attribute
_AVC_N_TIMES = 3


@register_args("AnnualValueCalculator")
def _(ctx):
    data_shape = _set_time_len(_AVC_N_TIMES, ctx, allow_one=False)
    return {
        "data_shape": data_shape,
        "timing": np.arange(0, _AVC_N_TIMES, dtype="datetime64[D]"),
    }


@register_args(
    [
        "AnnualValueCalculator._split_values_by_year",
        "AnnualValueCalculator.get_annual_means",
        "AnnualValueCalculator.get_annual_totals",
    ]
)
def _(ctx):
    data_shape = _set_time_len(_AVC_N_TIMES, ctx, allow_one=False)
    return {"values": np.ones(data_shape)}


# This has a coefficient dictionary that needs defining
register_args("calculate_kattge_knorr_arrhenius_factor")(
    lambda _: {"coef": {"ha": 1, "hd": 1, "entropy_intercept": 1, "entropy_slope": 1}}
)

# AcclimationModel
# Datetimes must be 1D
register_args(["AcclimationModel", "AcclimationModel._validate_and_set_datetimes"])(
    lambda _: {"datetimes": np.arange(0, _SUBDAILY_N_TIMES, dtype="datetime64[h]")}
)
# Subdaily -> daily methods must have length of first dim = number of times
register_args(
    [
        "AcclimationModel.get_daily_means",
        "AcclimationModel.get_window_values",
    ]
)(lambda ctx: {"values": np.ones((_SUBDAILY_N_TIMES, *ctx.shape[1:]))})
# Daily -> subdaily methods must have length of first dim = number of days
register_args(
    [
        "AcclimationModel.fill_daily_to_subdaily",
        "AcclimationModel._get_subdaily_interpolation_xy",
    ]
)(lambda ctx: {"values": np.ones((ceil(_SUBDAILY_N_TIMES / 24), *ctx.shape[1:]))})
# The automatic argument generation doesn't work for timedelta
register_args("AcclimationModel.set_nearest")(
    lambda _: {"time": np.timedelta64(12, "h")}
)


## Splash module
# The size of the time dimension needs to match the number of dates in the Calendar
_SPLASH_N_DATES = 10

register_args("SplashModel.estimate_daily_water_balance")(
    lambda ctx: {"previous_wn": np.full(_set_time_len(_SPLASH_N_DATES, ctx), 10)}
)
# wn_init should not include the time dimension of the shape
register_args("SplashModel.calculate_soil_moisture")(
    lambda ctx: {"wn_init": np.full(_del_time_dim(ctx), 10)}
)


@register_args("SplashModel")
def _(ctx):
    if ctx.parents and ctx.parents[0] == "SplashModel.estimate_initial_soil_moisture":
        # This requires at least 1 year of data
        n_dates = 366
    else:
        n_dates = _SPLASH_N_DATES
    shape = _set_time_len(n_dates, ctx)
    return {
        "dates": Calendar(np.arange(0, n_dates, dtype="datetime64[D]")),
        "lat": np.full(shape, 10),
        "elv": np.full(shape, 10),
        "sf": np.full(shape, 0.5),
        "tc": np.full(shape, 25),
        "pn": np.full(shape, 10),
    }


# DailySolarFluxes / DailyEvapFluxes
# The size of the first dimension needs to match the number of dates in the Calendar
_DAILY_FLUXES_N_DATES = 4


@register_args("DailySolarFluxes")
def _(ctx):
    shape = _set_time_len(_DAILY_FLUXES_N_DATES, ctx)
    return {
        "dates": Calendar(np.arange(0, _DAILY_FLUXES_N_DATES, dtype="datetime64[D]")),
        "latitude": np.full(shape, 10),
        "elevation": np.full(shape, 10),
        "sunshine_fraction": np.full(shape, 0.5),
        "temperature": np.full(shape, 25),
    }


@register_args("DailyEvapFluxes")
def _(ctx):
    shape = _set_time_len(_DAILY_FLUXES_N_DATES, ctx)
    return {
        "pa": np.full(shape, 10),
        "tc": np.full(shape, 25),
        "kWm": np.full(shape, 150),
    }


@register_args("DailyEvapFluxes.estimate_aet")
def _(ctx):
    shape = _set_time_len(_DAILY_FLUXES_N_DATES, ctx)
    return {"wn": np.full(shape, 10)}


## Phenology module

_PHENOLOGY_N_TIMES = 2

register_args("FaparLimitation")(
    lambda ctx: {"years": np.ones(ctx.bcast_shape[0], dtype="datetime64[Y]")}
)
register_args("FaparLimitation.from_pmodel")(
    lambda ctx: {
        "datetimes": np.arange(0, _PHENOLOGY_N_TIMES, dtype="datetime64[D]"),
        "aridity_index": np.ones(_set_time_len(1, ctx)),  # Time: constant or years (1)
    }
)
register_args("Phenology")(
    lambda _: {
        "daily_gpp": np.full((_PHENOLOGY_N_TIMES,), 0.5),
        "datetimes": np.arange(0, _PHENOLOGY_N_TIMES, dtype="datetime64[D]"),
        "fapar_limitation": FaparLimitation(
            annual_total_potential_gpp=np.ones(_PHENOLOGY_N_TIMES),
            annual_mean_ca=np.ones(_PHENOLOGY_N_TIMES),
            annual_mean_chi=np.ones(_PHENOLOGY_N_TIMES),
            annual_mean_vpd=np.ones(_PHENOLOGY_N_TIMES),
            annual_total_precip=np.ones(_PHENOLOGY_N_TIMES),
            annual_growing_season_length=np.ones(_PHENOLOGY_N_TIMES),
            aridity_index=np.ones(_PHENOLOGY_N_TIMES),
            years=np.zeros((_PHENOLOGY_N_TIMES,), dtype="datetime64[Y]"),
            phenology_const=PhenologyConst(
                z=12.227, k=0.5, f0_coefficients=(0.65, 0.604169, 1.9), sigma=0.771
            ),
        ),
    }
)

## Demography module
# This uses 1D arrays. These could probably be skipped instead.
# Inputs need the same number of PFTs / heights and PFT names.
_N_PFT = 3
_N_HEIGHTS = 2
_PFT_NAMES = [f"Tree{i + 1}" for i in range(_N_PFT)]

register_args("Cohorts")(
    lambda _: {
        "dbh_values": np.full(_N_PFT, 2),
        "n_individuals": np.ones(_N_PFT),
        "pft_names": np.array(_PFT_NAMES, dtype=np.str_),
    }
)
register_args("Flora")(
    lambda _: {"pfts": [PlantFunctionalType(name=name) for name in _PFT_NAMES]}
)
register_args("Flora.get_stem_traits")(lambda _: {"pft_names": _PFT_NAMES})
register_args("Canopy")(lambda _: {"fit_ppa": True})
register_args("CohortCanopyData")(
    lambda _: {
        "projected_leaf_area": np.ones((_N_HEIGHTS, _N_PFT)),
        "n_individuals": np.ones(_N_PFT),
        "lai": np.ones(_N_PFT),
        "par_ext": np.ones(_N_PFT),
    }
)
register_args("CommunityCanopyData")(
    lambda _: {
        "absorption": np.full((_N_HEIGHTS, _N_PFT), 0.5),
        "leaf_area_index": np.full((_N_HEIGHTS, _N_PFT), 0.5),
        "cohort_leaf_area": np.full((_N_HEIGHTS, _N_PFT), 1),
    }
)
register_args("StemAllometry")(lambda _: {"at_dbh": np.full(_N_PFT, 0.5)})
register_args("StemAllocation")(lambda _: {"whole_crown_gpp": np.full(_N_PFT, 0.5)})
register_args("CrownProfile")(
    lambda _: {"z": np.linspace(5, 15, _N_HEIGHTS)[:, np.newaxis]}
)
register_args(
    [
        "Cohorts.drop_cohort_data",
        "StemAllometry.drop_cohort_data",
        "Community.drop_cohorts",
    ]
)(lambda _: {"drop_indices": [0, 1]})
