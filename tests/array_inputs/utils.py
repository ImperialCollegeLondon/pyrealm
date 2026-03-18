"""Contains utility functions used in the tests for array inputs.

The structure of the tests are:
- Iterate through the methods / functions of the library.
- Generate input arguments according to their type hint, with the shape and type of
  array arguments defined. This may be called recursively for some types.
- Check if the function is a class method, if so instantiate the class using the
  same approach.
- Call the function twice with different (but equivalent) array inputs and check the
  result is the same (but not necessarily the shape).
- For class methods also check the class attributes are equivalent.

The functions / objects in this file are broadly split into four sections. With the main
functions listed below:
1. To get the list of functions / methods in the library.
   - `get_method_list`
2. To get the argument datatypes from the annotations. These are used in section 3.
3. To initialise function arguments and classes.
   - `generate_args`
   - `initialise_class`
4. To compare the function outputs / class attributes.
   - `assert_is_equal`
   - `compare_instances`

The functions that are only used internally in this module are marked private with a
leading underscore.
"""

from collections.abc import Callable, Iterator
from dataclasses import InitVar
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
import xarray as xr
from numpy.typing import DTypeLike

import pyrealm
from pyrealm.constants import ConstantsClass
from pyrealm.demography.flora import (
    PlantFunctionalType,
    PlantFunctionalTypeStrict,
    StemTraits,
)
from tests.array_inputs.context import Context
from tests.array_inputs.overrides import (
    ADDITIONAL_INIT_METHODS,
    IGNORE_OUTPUTS,
    MANUAL_ARGS,
    REQUIRES,
    SKIP_METHODS,
)


class _Config:
    DEBUG: bool = False


config = _Config()


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


def get_method_list(array_type: str) -> list[tuple[str, Callable, type | None]]:
    """Get a list of callables that take array inputs in the Pyrealm package.

    Args:
        array_type: The type of array inputs to look for. "numpy" or "ArrayType".

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

            if _has_array_input(method, array_type) and name not in SKIP_METHODS:
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


def _is_numpy_type(typ: Any) -> bool:
    """Returns True if the type is a numpy array. Prefer _is_array_type."""
    origin = get_origin(typ)  # Get the unannotated type, i.e. X[...] -> X

    try:
        # Handle annotated types like NDArray[np.float32]
        if origin is not None:
            return issubclass(origin, np.ndarray)

        # Handle basic types
        return issubclass(typ, np.ndarray)

    except TypeError:
        return False


def _is_array_type(typ: type, array_type: str) -> bool:
    """Returns True if the type is a numpy array or ArrayType."""
    typ = _strip_wrapped_types(typ)
    origin = get_origin(typ)  # Get the unannotated type, i.e. X[...] -> X

    # If Union[...] or X | Y then convert to an array of argument types
    if origin in (Union, UnionType):
        args = get_args(typ)
    else:
        args = (typ,)

    # Check for the different argument types
    has_numpy = False
    has_xarray = False
    for arg in args:
        if _is_numpy_type(arg):
            has_numpy = True

        elif arg is xr.DataArray:
            has_xarray = True

    # Determine if it has the correct argument types for the given 'array_type'
    if array_type == "numpy":
        if has_numpy:
            return True
    elif array_type == "ArrayType":
        if has_numpy and has_xarray:
            return True
    else:
        raise ValueError("Invalid array_type. Use 'numpy' or 'ArrayType'.")

    return False


def _has_array_input(method: Callable, array_type: str) -> bool:
    """Returns True if any of the method arguments are a numpy array."""
    from typing import get_type_hints

    try:
        hints = get_type_hints(method)
        hints = {k: v for k, v in hints.items() if k != "return"}
    except NameError:
        return False

    return any(_is_array_type(typ, array_type) for typ in hints.values())


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


def _get_manual_args(param_name: str, ctx: Context) -> Any:
    """Simplify call to MANUAL_ARGS."""
    if ctx.name not in MANUAL_ARGS:
        return None
    return MANUAL_ARGS[ctx.name](ctx).get(param_name)


def _get_parameters(
    method: Callable, ctx: Context
) -> dict[str, tuple[Parameter, type, str]]:
    """Get a dictionary of {parameter_name: (parameter, type, approach)}.

    Gets the relevant parameters using `inspect.signature().parameters`, as well as any
    specified by `REQUIRES`. Then uses `get_type_hints()` to ensure types are resolved.

    The 'approach' output describes how the argument will be defined. It is either
    "manual", "default", or "automatic".
    """
    from typing import get_type_hints

    # Get the method parameters with unnecessary arguments removed
    params = {
        name: p
        for name, p in signature(method).parameters.items()
        if not (name == "self" or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD))
    }

    # Add required keyword arguments specified in REQUIRES
    required_args = REQUIRES.get((ctx.name, tuple(ctx.parents)))
    if required_args is not None:
        params.update(required_args)

    # Check all annotations
    for param_name, param in params.items():
        if param.annotation == param.empty:
            raise Exception(f"Missing annotation for {ctx.name}:{param_name}")

    # Resolve string annotations using get_type_hints
    # Fall back to param.annotation for required keyword arguments
    type_hints = {
        name: get_type_hints(method).get(name, param.annotation)
        for name, param in params.items()
    }

    approaches: dict[str, str] = {}
    for name, param in params.items():
        if _get_manual_args(name, ctx) is not None:
            approaches[name] = "manual"
        elif param.default is not param.empty and not (
            required_args and name not in required_args  # Don't use default if REQUIRES
        ):
            approaches[name] = "default"
        else:
            approaches[name] = "automatic"

    return {name: (params[name], type_hints[name], approaches[name]) for name in params}


def _initialise_type_default(typ: Any, ctx: Context) -> Any:
    """Define the default value for each type."""
    from collections.abc import Sequence
    from random import randint
    from typing import TypeAliasType, get_origin

    from pyrealm.demography.flora import Flora

    # Handle basic wrapped types
    typ = _strip_wrapped_types(typ)

    # Handle TypeAliasType
    if isinstance(typ, TypeAliasType):
        typ = typ.__value__

    origin = get_origin(typ)
    if isinstance(origin, TypeAliasType):
        typ = origin.__value__

    # If Sequence[T]: create a list of 2 objects
    origin = get_origin(typ)
    args = get_args(typ)
    if origin is Sequence:
        inner_type = args[0] if args else Any
        return [_initialise_type_default(inner_type, ctx) for _ in range(2)]

    # If Union[...] or X | Y: Create an array if an option, otherwise use the first type
    if origin in (Union, UnionType):
        for arg in args:
            if _is_array_type(arg, "numpy"):
                return _initialise_type_default(arg, ctx)
        return _initialise_type_default(args[0], ctx)

    pft_names = ["Tree1", "Tree2", "Tree3"]

    # Numpy arrays
    if _is_array_type(typ, "numpy"):
        dtype = _extract_numpy_dtype(typ)
        shape = ctx.shape
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

    kwargs: dict[str, Any] = {}

    params = _get_parameters(method, ctx)

    # Get the number of array arguments for selecting array shapes / dimensions
    ctx.n_array = sum(
        _is_array_type(typ, "numpy") and approach != "default"
        for _, typ, approach in params.values()
    )
    ctx.i_array = -1

    for param_name, (param, typ, approach) in params.items():
        if _is_array_type(typ, "numpy"):
            ctx.i_array += 1

        # Set manually defined values
        if approach == "manual":
            kwargs[param_name] = _get_manual_args(param_name, ctx)

        # Set default arguments
        elif approach == "default":
            kwargs[param_name] = param.default
            continue  # Keep default array types

        # Initialise any other arguments
        elif approach == "automatic":
            kwargs[param_name] = _initialise_type_default(typ, ctx)

            # Adjust values where np.ones causes an issue
            match param_name:
                case "tk":
                    kwargs[param_name] += 273.15

        # If using a different array type, convert numpy arrays to this
        arg = kwargs[param_name]
        if (
            _is_array_type(typ, "ArrayType")
            and isinstance(arg, np.ndarray)
            and arg.ndim == len(ctx.shape)
        ):
            if ctx.array_type == "xarray":
                dims = ctx.array_dims or None
                kwargs[param_name] = xr.DataArray(arg, dims=dims)

    # Print the arguments if in debug mode
    if config.DEBUG:
        print(f"\n{ctx.name}\nParents: {ctx.parents}\nArguments:")
        for param_name, (_, _, approach) in params.items():
            print(f"\t{param_name}: ({approach})")
            arg = kwargs[param_name]
            if isinstance(arg, np.ndarray):
                out = f"ndarray({arg.shape}, dtype={arg.dtype})"
            elif isinstance(arg, xr.DataArray):
                out = f"DataArray({arg.shape}, dims={arg.dims}, dtype={arg.dtype})"
            elif (  # Replace pyrealm objects with a multiline repr with just the name
                getattr(arg.__class__, "__module__", "").startswith("pyrealm")
                and "\n" in repr(arg)
            ) or isinstance(arg, ConstantsClass):
                out = arg.__class__.__name__
            else:
                out = repr(arg)
            print("\t\t" + out.replace("\n", "\n\t\t"))

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
def _is_equal(val1: Any, val2: Any, broadcast: bool = False) -> bool:
    """Compare if two variables are equal. Optionally, broadcast to same shape."""

    if type(val1) is not type(val2):
        return False

    if hasattr(val1, "__array__"):
        if broadcast:
            val1, val2 = np.broadcast_arrays(val1, val2)

        if np.issubdtype(val1.dtype, np.str_):
            return np.array_equal(val1, val2)
        else:
            return np.array_equal(val1, val2, equal_nan=True)

    elif isinstance(val1, list | tuple) and isinstance(val2, list | tuple):
        if len(val1) != len(val2):
            return False
        return all(_is_equal(v1, v2, broadcast) for v1, v2 in zip(val1, val2))

    elif hasattr(val1, "__dict__") and hasattr(val2, "__dict__"):
        compare_instances(val1, val2, broadcast)  # Raises if not equal
        return True

    else:
        return val1 == val2


def _comparison_string(val1: Any, val2: Any) -> str:
    """Returns a string representation of two variables that are not equal."""

    def value_string(val: Any) -> str:
        if hasattr(val, "__array__") and val.size > 5:
            val_str = f"<array> {val.shape}"
        else:
            val_str = str(val).replace("\n", " ")
        val_str = val_str[:30] + ".." if len(val_str) > 30 else val_str
        return val_str

    return value_string(val1) + " != " + value_string(val2)


def assert_is_equal(val1: Any, val2: Any, raise_msg: str, broadcast: bool = False):
    """Raise if two variables are not equal. Optionally, broadcast to same shape."""

    if not _is_equal(val1, val2, broadcast):
        attr_comparison = _comparison_string(val1, val2)
        raise ValueError(f"{raise_msg} ({attr_comparison})")


def compare_instances(instance1: Any, instance2: Any, broadcast: bool = False):
    """Raises ValueError if the two class instances do not have equal attributes.

    Set `broadcast=True` to broadcast attributes to a common shape for comparison.

    If broadcasting, this function ignores the 'shape' attribute of any class, which is
    not expected to broadcast, and anything in the manually defined list IGNORE_OUTPUTS.
    Any 'dims' attributes will also be ignored.
    """
    dict1 = instance1.__dict__
    dict2 = instance2.__dict__
    class_name = instance1.__class__.__name__
    for key in dict1:
        if broadcast:
            if key == "shape":
                continue
            if f"{class_name}:{key}" in IGNORE_OUTPUTS:
                continue
        if key == "dims":
            continue

        raise_msg = f"{class_name}: {key} not equal"
        assert_is_equal(dict1[key], dict2[key], raise_msg, broadcast)
