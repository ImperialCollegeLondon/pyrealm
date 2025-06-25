"""This module contains the tests to check broadcastable array inputs.

This ensures that the outputs/attributes of any functions/methods are unchanged when
broadcastable array inputs are used in place of the full size arrays.
"""

import dataclasses
import inspect
import warnings
from collections.abc import Callable, Iterator
from types import ModuleType
from typing import Any, Union, cast

import numpy as np
import numpy.typing as npt

import pyrealm
from pyrealm.core.calendar import Calendar
from pyrealm.core.experimental import ExperimentalFeatureWarning
from pyrealm.demography.flora import (
    PlantFunctionalType,
    PlantFunctionalTypeStrict,
    StemTraits,
)

warnings.filterwarnings("ignore", category=ExperimentalFeatureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# These methods are not relevant or are incompatible without additional work
skip_methods = [
    # PModel
    "AcclimationModel.set_include",
    "PModelABC",  # Cannot init ABC
    "QuantumYieldABC",  # Cannot init ABC
    "OptimalChiABC.estimate_chi",  # Cannot init ABC
    "OptimalChiC4RootzoneStress.estimate_chi",  # Requires rootzonestress in PME
    "OptimalChiC4NoGammaRootzoneStress.estimate_chi",  # Requires rootzonestress in PME
    "OptimalChiPrentice14RootzoneStress.estimate_chi",  # Requires rootzonestress in PME
    "OptimalChiLavergne20C3.estimate_chi",  # Requires theta in PME
    "OptimalChiLavergne20C4.estimate_chi",  # Requires theta in PME
    "QuantumYieldSandoval",  # Requires aridity_index in PME
    "QuantumYieldSandoval.peak_quantum_yield",  # Requires aridity_index in PME
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


# These methods require specific arguments
def method_args(argument: str, ctx: "Context"):
    """Return any manually defined arguments for the current callable in the context."""
    shape = ctx.shape()

    # PModel parameters
    splashDatesLen = 10
    # Demography parameters
    n_pft = 3
    n_heights = 2
    pft_names = [f"Tree{i + 1}" for i in range(n_pft)]

    method_arguments_list: dict[str, dict] = {
        ## PModel
        # Subdaily data needs more than 1 day of times (uses 48 hours)
        "AcclimationModel": {"datetimes": np.arange(0, 48, dtype="datetime64[h]")},
        "AcclimationModel.set_nearest": {"time": np.timedelta64(12, "h")},
        "AcclimationModel._validate_and_set_datetimes": {
            "datetimes": np.arange(0, 48, dtype="datetime64[h]")
        },
        "AcclimationModel._get_subdaily_interpolation_xy": {"values": np.ones(2)},
        "AcclimationModel.fill_daily_to_subdaily": {"values": np.ones(2)},
        "AcclimationModel.get_window_values": {"values": np.ones(48)},
        "AcclimationModel.get_daily_means": {"values": np.ones(48)},
        "calculate_kattge_knorr_arrhenius_factor": {
            "coef": {"ha": 1, "hd": 1, "entropy_intercept": 1, "entropy_slope": 1}
        },
        "SplashModel.estimate_daily_water_balance": {
            "previous_wn": np.full(shape[1:], splashDatesLen),
        },
        "SplashModel.calculate_soil_moisture": {"wn_init": np.full(shape[1:], 10)},
        "PModelEnvironment": {"patm": np.full(shape, 100000)},
        "TwoLeafIrradiance": {"patm": np.full(shape, 100000)},
        ## Demography uses 1D arrays
        "Cohorts": {
            "dbh_values": np.zeros(n_pft),
            "n_individuals": np.ones(n_pft),
            "pft_names": np.array(pft_names, dtype=np.str_),
        },
        "Flora": {"pfts": [PlantFunctionalType(name=name) for name in pft_names]},
        "Flora.get_stem_traits": {"pft_names": pft_names},
        "Canopy": {"fit_ppa": True},
        "CohortCanopyData": {
            "projected_leaf_area": np.ones((n_heights, n_pft)),
            "n_individuals": np.ones(n_pft),
            "pft_lai": np.ones(n_pft),
            "pft_par_ext": np.ones(n_pft),
        },
        "CommunityCanopyData": {"cohort_transmissivity": np.ones((n_heights, n_pft))},
        "StemAllometry": {"at_dbh": np.full(n_pft, 0.5)},
        "StemAllocation": {"whole_crown_gpp": np.full(n_pft, 0.5)},
        "CrownProfile": {"z": np.linspace(5, 15, n_heights)[:, np.newaxis]},
    }
    arguments: dict = method_arguments_list.get(ctx.name, {})

    # Arguments that use temporary variables or depend on parents
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
additional_init_methods = {
    "AcclimationModel": "set_nearest",
}


## Functions to get the list of callables
def get_package_modules(pkg: ModuleType) -> list[ModuleType]:
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


def get_module_callables(
    module: ModuleType,
) -> Iterator[tuple[str, Callable, type | None]]:
    """Get the callables contained within a module.

    Returns an iterable including the function/method name, callable, and (if a method)
    class.
    """
    for name, obj in inspect.getmembers(module):
        if getattr(obj, "__module__", None) != module.__name__:
            continue
        if inspect.isfunction(obj) or inspect.ismethod(obj):
            yield name, obj, None
        elif inspect.isclass(obj):
            for mname, method in inspect.getmembers(obj, predicate=inspect.isfunction):
                if mname in ["__init__", "__post_init__"]:
                    yield name, method, obj
                else:
                    yield f"{name}.{mname}", method, obj


def is_instance_method(cls: type | None, method_name: str) -> bool:
    """Returns True if the method is not static or a classmethod."""
    if cls is None:
        return False
    attr = inspect.getattr_static(cls, method_name)
    if isinstance(attr, staticmethod | classmethod):
        return False
    else:
        return True


## Functions to get the argument datatypes
def is_array_type(typ: Any) -> bool:
    """Returns True if the type is a numpy array."""
    from dataclasses import InitVar
    from types import UnionType
    from typing import get_args, get_origin

    origin = get_origin(typ)  # Get the unannotated type, i.e. X from X[...]

    # Handle Union[...] or X | Y
    if origin in (Union, UnionType):
        return any(is_array_type(arg) for arg in get_args(typ))

    # Handle dataclasses.InitVar[...]
    if isinstance(typ, InitVar):
        return is_array_type(typ.type)

    try:
        # Handle annotated types like NDArray[np.float32]
        if origin is not None:
            return issubclass(origin, np.ndarray)

        # Handle basic types
        return issubclass(typ, np.ndarray)

    except TypeError:
        return False


def has_array_input(method: Callable) -> bool:
    """Returns True if any of the method arguments are a numpy array."""
    from typing import get_type_hints

    try:
        hints = get_type_hints(method)
        hints = {k: v for k, v in hints.items() if k != "return"}
    except NameError:
        return False

    return any(is_array_type(typ) for typ in hints.values())


def extract_numpy_dtype(typ: Any) -> npt.DTypeLike:
    """Extract a numpy dtype from NDArray annotation."""
    from typing import get_args, get_origin

    dtype: npt.DTypeLike = np.float64

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
@dataclasses.dataclass
class Context:
    """Context class to pass between functions.

    Used to initialise of arguments that depend on array shapes and the heirarchical
    function/argument class structure.

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
    parents: list[str] = dataclasses.field(default_factory=list)
    """A list of the superior function / classes for the current context."""

    def new(self, name: str) -> "Context":
        """Generate a context for a new function/class, updating the heirarchy."""
        return Context(name, self.shapes, self.i_arg, [*self.parents, self.name])

    def shape(self) -> tuple[int, ...]:
        """Return the shape for index `i_arg`."""
        return self.shapes[self.i_arg % len(self.shapes)]


def initialise_type_default(typ: Any, ctx: Context) -> Any:
    """Define the default value for each type."""
    from collections.abc import Sequence
    from dataclasses import InitVar
    from random import randint
    from types import UnionType
    from typing import get_args, get_origin

    from pyrealm.demography.flora import Flora

    # Handle any wrapped types
    # InitVar[T]
    if isinstance(typ, InitVar):
        typ = typ.type
    # Sequence[T] (create a sequence of 2)
    origin = get_origin(typ)
    args = get_args(typ)
    if origin is Sequence:
        inner_type = args[0] if args else Any
        return [initialise_type_default(inner_type, ctx) for _ in range(2)]
    # Type[T]
    if origin is type:
        return initialise_type_default(args[0], ctx)
    # Union[...] or X | Y
    if origin in (Union, UnionType):
        # Use an array if an option
        for arg in args:
            if is_array_type(arg):
                return initialise_type_default(arg, ctx)
        # Otherwise first type in list
        return initialise_type_default(args[0], ctx)

    pft_names = ["Tree1", "Tree2", "Tree3"]

    # Numpy arrays
    if is_array_type(typ):
        dtype = extract_numpy_dtype(typ)
        shape = ctx.shape()
        if np.issubdtype(dtype, np.datetime64):
            return np.full(shape, np.datetime64("2000-01-01"), dtype=dtype)
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
        return initialise_type_default(Flora, ctx).get_stem_traits(pft_names)
    elif len(inspect.signature(typ).parameters) > 0:
        return initialise_class(typ, ctx)
    else:
        return typ()


def generate_args(method: Callable, ctx: Context) -> dict[str, Any]:
    """Generate the arguments needed for a function.

    Requires type hinting. Numpy arrays are defined using the shapes argument.
    """
    from typing import get_type_hints

    kwargs = {}
    sig = inspect.signature(method)
    ctx.i_arg = 0
    for param_name, param in sig.parameters.items():
        ctx.i_arg += 1

        # Set manually defined values
        manual_arg = method_args(param_name, ctx)
        if manual_arg is not None:
            kwargs[param_name] = manual_arg

        # Skip unnecesary arguments
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
            # Get the contexts of the pyrealm classes to pass to get_type_hints
            globalns = {}
            for module in get_package_modules(pyrealm):
                globalns.update(vars(module))
            # Resolve any string annotations
            typ = get_type_hints(method, globalns=globalns).get(
                param_name, param.annotation
            )
            kwargs[param_name] = initialise_type_default(typ, ctx)

    return kwargs


def initialise_class(cls: type, ctx: Context) -> Any:
    """Initialise class input arguments and then the class."""
    name = cls.__name__
    ctx_class = ctx.new(name)
    args = generate_args(cls.__init__, ctx_class)  # type: ignore[misc]
    instance = cls(**args)
    # If there are any additional methods required for initialisation call these
    if name in additional_init_methods:
        mname = additional_init_methods[name]
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

    elif isinstance(val1, tuple) and isinstance(val2, tuple):
        if len(val1) != len(val2):
            return False
        return all(is_equal(v1, v2) for v1, v2 in zip(val1, val2))

    elif hasattr(val1, "__dict__") and hasattr(val2, "__dict__"):
        return compare_instances(val1, val2)

    else:
        return val1 == val2


def compare_instances(instance1: Any, instance2: Any) -> bool:
    """Compare if two class instances have equal attributes."""
    dict1 = instance1.__dict__
    dict2 = instance2.__dict__
    for key in dict1:
        if not is_equal(dict1[key], dict2[key]):
            print(f"{instance1.__class__.__name__}: {key} not equal")
            # print(dict1[key])
            # print(dict2[key])
            # raise ValueError
            return False
    return True


shapes: list[tuple[int, ...]]
shape_full: list[tuple[int, ...]]
shapes = [(3, 2, 2), (1, 2, 2), (3, 1, 1), (1, 1, 1)]
# shapes = [(1, 1, 1)]
shape_full = [(3, 2, 2)]

for mod in get_package_modules(pyrealm):
    for name, method, cls in get_module_callables(mod):
        if not has_array_input(method) or name in skip_methods:
            continue

        is_correct = True

        # Generate the arguments for the method
        ctx = Context(name, shapes)
        ctx_full = Context(name, shape_full)
        args = generate_args(method, ctx)
        args_full = generate_args(method, ctx_full)

        if is_instance_method(cls, method.__name__):
            cls = cast(type, cls)  # Make mypy aware this cannot be None
            # First initialise class and get bound methods
            instance1 = initialise_class(cls, ctx)
            instance2 = initialise_class(cls, ctx_full)
            method1 = getattr(instance1, method.__name__)
            method2 = getattr(instance2, method.__name__)
            # Run the method
            result = method1(**args)
            result_full = method2(**args_full)
            # Compare results
            is_correct = is_equal(result, result_full)
            if not is_correct:
                print(name, ": Result not equal")
            is_correct = is_equal(result, result_full) and compare_instances(
                instance1, instance2
            )

        else:
            # Run the method
            result = method(**args)
            result_full = method(**args_full)
            # Compare results
            is_correct = is_equal(result, result_full)
            if not is_correct:
                print(name, ": Result not equal")

        # if (not is_correct):
        #     print(name)
        #     print()
        # # Compare the results to see if they are identical
        # if (not is_correct):
        # raise Error(f'Results do not match in {name}')
