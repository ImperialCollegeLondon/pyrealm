"""Contains utility functions used in test_broadcasting.py.

The structure of the test is:
- Iterate through the methods / functions of the library.
- Generate input arguments according to their type hint, with the shape of array
  arguments defined. This may be called recursively for some types.
- Check if the function is a class method, if so instantiate the class using the
  same approach.
- Call the function once with broadcastable array shapes and once with the fully
  broadcast inputs. And check the result is the same (but not necessarily the shape).
- For class methods also check the class attributes are equivalent.

The functions / objects in this file are broadly split into five sections. With the main
functions listed below:
1. To manually define function arguments / which functions to ignore, etc in order to
   fix errors.
   - `SKIP_METHODS`
   - `IGNORE_OUTPUTS`
   - `REQUIRES`
   - `ADDITIONAL_INIT_METHODS`
   - `defined_method_args`
2. To get the list of functions / methods in the library.
   - `get_method_list`
3. To get the argument datatypes from the annotations. These are used in section 4.
4. To initialise function arguments and classes.
   - `generate_args`
   - `initialise_class`
   - `Context` - To keep track of the array shapes and when checking values in section 1
5. To compare the function outputs / class attributes.
   - `is_equal`
   - `compare_instances`

The functions that are not used in `test_broadcasting` and are only used within this
file are marked private.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import InitVar, dataclass, field
from inspect import (
    Parameter,
    getattr_static,
    getmembers,
    isabstract,
    isclass,
    isfunction,
    ismethod,
    signature,
)
from types import ModuleType, UnionType
from typing import Any, Union, get_args, get_origin

import numpy as np
from numpy.typing import DTypeLike, NDArray

import pyrealm
from pyrealm.core.calendar import Calendar
from pyrealm.demography.flora import (
    PlantFunctionalType,
    PlantFunctionalTypeStrict,
    StemTraits,
)
from pyrealm.phenology.fapar_limitation import FaparLimitation, PhenologyConst

## Lists / functions to manually define arguments, methods or outputs to ignore, etc.

# These methods are not relevant or are incompatible without additional work
SKIP_METHODS = [
    "evaluate_horner_polynomial",  # Coefficients are 1D
    # PModel
    "AcclimationModel.set_include",
    "PModel._get_daily_gpp",
    # The following two take an array that needs to be congruent with the existing
    # PModel.env shape.
    "PModel.apply_gpp_penalty_factor",
    "SubdailyPModel.apply_gpp_penalty_factor",
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
    "calculate_foliage_masses",
    "calculate_fine_root_masses",
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


# Ignore these outputs, they are not expected to be equal.
# Formats: [fn name] for function results, [class]:[attr] for class attributes
IGNORE_OUTPUTS = [
    "Cohorts:_cohort_id",
    "Calendar:n_dates",
]

# The REQUIRES dictionary provides a way of populating required keywords arguments for
# some methods. The keys specify the object being created and the parent context in
# which it is created and the values provide a tuple of arguments that should be added
# to the method signature.


def kwarg_params(names: tuple[str, ...]) -> dict[str, Parameter]:
    """Creates a dictionary containing inspect.Parameter instances."""
    return {
        name: Parameter(
            name=name,
            kind=Parameter.POSITIONAL_OR_KEYWORD,
            annotation=NDArray[np.floating],
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
    ): kwarg_params(("rootzonestress",)),
    (
        "PModelEnvironment",
        (
            "OptimalChiC4NoGammaRootzoneStress.estimate_chi",
            "OptimalChiC4NoGammaRootzoneStress",
        ),
    ): kwarg_params(("rootzonestress",)),
    (
        "PModelEnvironment",
        (
            "OptimalChiPrentice14RootzoneStress.estimate_chi",
            "OptimalChiPrentice14RootzoneStress",
        ),
    ): kwarg_params(("rootzonestress",)),
    (
        "PModelEnvironment",
        (
            "OptimalChiLavergne20C3.estimate_chi",
            "OptimalChiLavergne20C3",
        ),
    ): kwarg_params(("theta",)),
    (
        "PModelEnvironment",
        (
            "OptimalChiLavergne20C4.estimate_chi",
            "OptimalChiLavergne20C4",
        ),
    ): kwarg_params(("theta",)),
    (
        "PModelEnvironment",
        ("QuantumYieldSandoval", "QuantumYieldSandoval"),
    ): kwarg_params(("aridity_index", "mean_growth_temperature")),
    (
        "PModelEnvironment",
        (
            "QuantumYieldSandoval.peak_quantum_yield",
            "QuantumYieldSandoval",
        ),
    ): kwarg_params(("aridity_index", "mean_growth_temperature")),
}


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
    shape = ctx.shape()
    bcast_shape = ctx.bcast_shape()

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
        "broadcast_time": {"shape": (3, *shape[1:])},
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
            envShape = (1 if shape[0] == 1 else 48, *shape[1:])
            arguments = {
                "tc": np.full(envShape, 20),
                "vpd": np.full(envShape, 40),
                "co2": np.full(envShape, 1000),
                "patm": np.full(envShape, 101325),
                "fapar": np.full(envShape, 1),
                "ppfd": np.full(envShape, 800),
            }

    return arguments.get(argument)


# Call additional methods when initialising these classes
ADDITIONAL_INIT_METHODS = {
    "AcclimationModel": "set_nearest",
}


## Functions to get the list of callables
def _get_package_modules(pkg: ModuleType) -> list[ModuleType]:
    """Get a list of modules contained within the package."""
    import importlib
    import pkgutil

    modules = []
    for _, modname, ispkg in pkgutil.walk_packages(
        pkg.__path__, prefix=pkg.__name__ + "."
    ):
        if not ispkg:
            modules.append(importlib.import_module(modname))

    return modules


def global_namespace():
    """Extract the global namespace for the package.

    Provides the contexts of the pyrealm classes to pass to get_type_hints.
    """

    globalns = {}
    for module in _get_package_modules(pyrealm):
        globalns.update(vars(module))

    return globalns


GLOBALNS = global_namespace()


def _is_instance_method(cls: type | None, method_name: str) -> bool:
    """Returns True if the method is not static or a classmethod."""
    if cls is None:
        return False
    attr = getattr_static(cls, method_name)
    if isinstance(attr, staticmethod | classmethod):
        return False
    else:
        return True


def _get_module_callables(
    module: ModuleType,
) -> Iterator[tuple[str, Callable, type | None]]:
    """Get the callables contained within a module.

    Returns:
        An iterable including the function/method name, callable, and (if a method)
        class.
    """
    for name, obj in getmembers(module):
        if getattr(obj, "__module__", None) != module.__name__:
            continue
        if isfunction(obj) or ismethod(obj):
            yield name, obj, None
        elif isclass(obj):
            for mname, method in getmembers(obj, predicate=isfunction):
                if mname in ["__init__", "__post_init__"]:
                    full_name = name
                else:
                    full_name = f"{name}.{mname}"

                if not _is_instance_method(obj, mname):
                    class_obj = None
                else:
                    class_obj = obj

                yield full_name, method, class_obj


def get_method_list() -> list[tuple[str, Callable, type | None]]:
    """Get a list of callables that take array inputs in the Pyrelam package.

    Returns:
        A list of callables, each containing the name ([function] or [class].[method]),
        the callable object, and the type of the class (for class instance methods).
    """
    method_list = []
    for mod in _get_package_modules(pyrealm):
        for name, method, cls in _get_module_callables(mod):
            # Don't add methods of abstract classes
            if cls is not None and isabstract(cls):
                attr = getattr_static(cls, name, None)
                if not isinstance(attr, staticmethod):
                    continue

            if _has_array_input(method) and name not in SKIP_METHODS:
                method_list.append((name, method, cls))
    return method_list


## Functions to get the argument datatypes
def _strip_wrapped_types(typ: Any) -> Any:
    """Handle basic wrapped types to get the inner type."""
    # InitVar[T] -> T
    if isinstance(typ, InitVar):
        return _strip_wrapped_types(typ.type)
    # Type[T] -> T
    if get_origin(typ) is type:
        args = get_args(typ)

        return _strip_wrapped_types(args[0])
    return typ


def _is_array_type(typ: Any) -> bool:
    """Returns True if the type is a numpy array."""
    typ = _strip_wrapped_types(typ)

    # If Union[...] or X | Y then check both types
    origin = get_origin(typ)  # Get the unannotated type, i.e. X[...] -> X
    if origin in (Union, UnionType):
        return any(_is_array_type(arg) for arg in get_args(typ))

    try:
        # Handle annotated types like NDArray[np.float32]
        if origin is not None:
            return issubclass(origin, np.ndarray)

        # Handle basic types
        return issubclass(typ, np.ndarray)

    except TypeError:
        return False


# Resolve issue with get_type_hints failing for InitVars in py3.10
# Define a stub to make InitVar callable (https://stackoverflow.com/questions/70400639)
InitVar.__call__ = lambda *args: None  # type: ignore[method-assign]


def _has_array_input(method: Callable) -> bool:
    """Returns True if any of the method arguments are a numpy array."""
    from typing import get_type_hints

    try:
        hints = get_type_hints(method)
        hints = {k: v for k, v in hints.items() if k != "return"}
    except NameError:
        return False

    return any(_is_array_type(typ) for typ in hints.values())


def _extract_numpy_dtype(typ: Any) -> DTypeLike:
    """Extract a numpy dtype from NDArray annotation."""
    dtype: DTypeLike = np.float64

    args = get_args(typ)
    if not args:
        return dtype  # If no annotation

    for arg in args:
        # args could be like: (tuple[int, ...], numpy.dtype[numpy.datetime64])
        if get_origin(arg) is np.dtype:
            dtype_args = get_args(arg)
            try:
                dtype = np.dtype(dtype_args[0]).type
            except TypeError:
                continue
        # Or like (np.float64)
        elif isinstance(arg, type) and issubclass(arg, np.generic):
            dtype = np.dtype(arg)

    # Use a default unit if a datetime
    if dtype == np.datetime64:
        dtype = np.dtype("datetime64[D]")

    return dtype


## Functions to initialise arguments and classes
@dataclass
class Context:
    """Context class to pass between functions.

    Used to initialise arguments that depend on array shapes or for manual overrides
    that rely upon the hierarchy of function/argument definitions.

    Attributes:
        name (str): Name of the current method/function/class.
        shapes (list[tuple[int, ...]]): Shapes to iterate over when generating array
            arguments.
        i_arg (int): Index of the current argument, used to select a shape.
        parents (list[str]): Hierarchy of names leading to the current function/class.
            The first value will be the name of the callable being tested. If it is a
            class method, the second value will be the name of the class.
    """

    name: str
    shapes: list[tuple[int, ...]]
    i_arg: int = 0
    parents: list[str] = field(default_factory=list)
    """A list of the superior function / classes for the current context."""

    def new(self, name: str) -> Context:
        """Generate a context for a new function/class, updating the hierarchy."""
        return Context(name, self.shapes, self.i_arg, [*self.parents, self.name])

    def shape(self) -> tuple[int, ...]:
        """Return the shape for index `i_arg`."""
        return self.shapes[self.i_arg % len(self.shapes)]

    def bcast_shape(self) -> tuple[int, ...]:
        """The broadcast shape of all inputs (not the full shape being tested)."""
        return np.broadcast_shapes(*self.shapes)


def _initialise_type_default(typ: Any, ctx: Context) -> Any:
    """Define the default value for each type."""
    from collections.abc import Sequence
    from random import randint

    from pyrealm.demography.flora import Flora

    # Handle basic wrapped types
    typ = _strip_wrapped_types(typ)

    # If Sequence[T]: create a list of 2 objects
    origin = get_origin(typ)
    args = get_args(typ)
    if origin is Sequence:
        inner_type = args[0] if args else Any
        return [_initialise_type_default(inner_type, ctx) for _ in range(2)]

    # If Union[...] or X | Y: Create an array if an option, otherwise use the first type
    if origin in (Union, UnionType):
        for arg in args:
            if _is_array_type(arg):
                return _initialise_type_default(arg, ctx)
        return _initialise_type_default(args[0], ctx)

    pft_names = ["Tree1", "Tree2", "Tree3"]

    # Numpy arrays
    if _is_array_type(typ):
        dtype = _extract_numpy_dtype(typ)
        shape = ctx.shape()
        if np.issubdtype(dtype, np.datetime64):
            return np.full(shape, 1, dtype="datetime64[D]")
        else:
            return np.ones(shape, dtype=dtype)

    # Other types
    elif typ is str:
        return ""
    elif typ is bool:
        return True
    elif typ is int:
        return 1
    elif typ is float:
        return 1
    elif typ is Any:
        return None
    elif typ is PlantFunctionalTypeStrict:
        return PlantFunctionalType(name=f"default.{randint(1, 10000)}")
    elif typ is Flora:
        return Flora([PlantFunctionalType(name=name) for name in pft_names])
    elif typ is StemTraits:
        return _initialise_type_default(Flora, ctx).get_stem_traits(pft_names)
    elif len(signature(typ).parameters) > 0:
        return initialise_class(typ, ctx)
    else:
        return typ()


def generate_args(method: Callable, ctx: Context) -> dict[str, Any]:
    """Generate the arguments needed for a function.

    Requires type hinting. Numpy arrays are defined using the shapes information in the
    Context.

    Args:
        method (Callable): The function or method to generate arguments for.
        ctx (Context): The context for shape and parent information.

    Returns:
        dict[str, Any]: The generated arguments for the function/method.
    """
    from typing import get_type_hints

    kwargs = {}

    # Get the method parameters and copy to get a modifiable OrderedDict from inside the
    # Parameters mappingproxy return type
    params = signature(method).parameters.copy()

    required_args = REQUIRES.get((ctx.name, tuple(ctx.parents)))

    if required_args is not None:
        params.update(required_args)

    ctx.i_arg = 0

    for param_name, param in params.items():
        ctx.i_arg += 1

        # Set manually defined values
        manual_arg = defined_method_args(param_name, ctx)
        if manual_arg is not None:
            kwargs[param_name] = manual_arg

        # Skip unnecessary arguments
        elif param_name == "self" or param.kind in (
            param.VAR_POSITIONAL,
            param.VAR_KEYWORD,
        ):
            continue

        # Set default arguments
        elif param.default is not param.empty:
            kwargs[param_name] = param.default

        # Initialise any other arguments
        else:
            if param.annotation is param.empty:
                raise Exception(f"Missing annotation for {ctx.name}:{param_name}")

            # Resolve any string annotations using the global namespace
            typ = get_type_hints(method, globalns=GLOBALNS).get(
                param_name, param.annotation
            )
            kwargs[param_name] = _initialise_type_default(typ, ctx)

            # Adjust values where np.ones causes an issue
            match param_name:
                case "tk":
                    kwargs[param_name] += 273.15

    return kwargs


def initialise_class(cls: type, ctx: Context) -> Any:
    """Initialise class input arguments and then the class.

    Args:
        cls (type): The class to initialise.
        ctx (Context): The context for shape and parent information.

    Returns:
        Any: The initialised class instance.
    """
    name = cls.__name__
    ctx_class = ctx.new(name)
    args = generate_args(cls.__init__, ctx_class)  # type: ignore[misc]
    instance = cls(**args)
    # If there are any additional methods required for initialisation call these
    if name in ADDITIONAL_INIT_METHODS:
        mname = ADDITIONAL_INIT_METHODS[name]
        method = getattr(instance, mname)
        args = generate_args(method, ctx_class.new(name + "." + mname))
        method(**args)
    return instance


## Functions to compare the results
def is_equal(val1: Any, val2: Any) -> bool:
    """Compare if two variables are equal."""
    if isinstance(val1, np.ndarray):
        if np.issubdtype(val1.dtype, np.str_):
            return np.array_equal(val1, val2)
        val1_b, val2_b = np.broadcast_arrays(val1, val2)
        equal = val1_b == val2_b
        both_nan = np.isnan(val1_b) & np.isnan(val2_b)
        return bool(np.all(equal | both_nan))

    elif isinstance(val1, list | tuple) and isinstance(val2, list | tuple):
        if len(val1) != len(val2):
            return False
        return all(is_equal(v1, v2) for v1, v2 in zip(val1, val2))

    elif hasattr(val1, "__dict__") and hasattr(val2, "__dict__"):
        compare_instances(val1, val2)  # Raises if not equal
        return True

    else:
        return val1 == val2


def comparison_string(val1: Any, val2: Any) -> str:
    """Returns a string representation of two variables that are not equal."""

    def value_string(val: Any) -> str:
        if isinstance(val, np.ndarray) and val.size > 5:
            val_str = f"<array> {val.shape}"
        else:
            val_str = str(val).replace("\n", " ")
        val_str = val_str[:30] + ".." if len(val_str) > 30 else val_str
        return val_str

    return value_string(val1) + " != " + value_string(val2)


def compare_instances(instance1: Any, instance2: Any):
    """Raises ValueError if the two class instances do not have equal attributes.

    This function ignores the shape attribute of any class, which is not expected to
    broadcast, and anything in the manually defined list IGNORE_OUTPUTS.
    """
    dict1 = instance1.__dict__
    dict2 = instance2.__dict__
    class_name = instance1.__class__.__name__
    for key in dict1:
        if key == "shape":
            continue
        if f"{class_name}:{key}" in IGNORE_OUTPUTS:
            continue
        if not is_equal(dict1[key], dict2[key]):
            attr_comparison = comparison_string(dict1[key], dict2[key])
            raise ValueError(f"{class_name}: {key} not equal ({attr_comparison})")
