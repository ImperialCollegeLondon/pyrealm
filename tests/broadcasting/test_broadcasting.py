import inspect

import numpy as np

import pyrealm
from pyrealm.demography.flora import PlantFunctionalType, PlantFunctionalTypeStrict


def get_package_modules(pkg):
    import importlib
    import pkgutil

    modules = []
    for _, modname, ispkg in pkgutil.walk_packages(
        pkg.__path__, prefix=pkg.__name__ + "."
    ):
        if not ispkg:
            modules.append(importlib.import_module(modname))
    return modules


def get_module_callables(module):
    """Returns an iterable including the function/method name, callable, and (if a method) class."""
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) or inspect.ismethod(obj):
            yield name, obj, None
        elif inspect.isclass(obj):
            for mname, method in inspect.getmembers(obj, predicate=inspect.isfunction):
                yield f"{name}.{mname}", method, obj


def is_instance_method(cls, name):
    if cls is None:
        return False
    attr = inspect.getattr_static(cls, name)
    if isinstance(attr, (staticmethod, classmethod)):
        return False
    else:
        return True


def is_array_type(typ):
    from dataclasses import InitVar
    from types import UnionType
    from typing import Union, get_args, get_origin

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


def has_array_input(method):
    from typing import get_type_hints

    try:
        hints = get_type_hints(method)
        hints = {k: v for k, v in hints.items() if k != "return"}
    except NameError:
        return False

    return any(is_array_type(typ) for typ in hints.values())


def extract_numpy_dtype(typ) -> np.dtype:
    """Extract a numpy dtype from NDArray annotation."""
    from typing import get_args, get_origin

    dtype = np.float64

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


def initialise_type_default(typ, shape):
    """Define the default value for each type."""
    from collections.abc import Sequence
    from dataclasses import InitVar
    from random import randint
    from typing import Any, get_args, get_origin

    # Handle any wrapped types
    # InitVar[T]
    if isinstance(typ, InitVar):
        typ = typ.type
    # Sequence[T] (create a sequence of 2)
    origin = get_origin(typ)
    args = get_args(typ)
    if origin is Sequence:
        inner_type = args[0] if args else Any
        return [initialise_type_default(inner_type, shape) for _ in range(2)]
    # Type[T]
    if origin is type:
        return initialise_type_default(args[0], shape)

    # Numpy arrays
    if is_array_type(typ):
        dtype = extract_numpy_dtype(typ)
        if dtype == np.datetime64:
            return np.full(shape, np.datetime64("2000-01-01"), dtype=dtype)
        else:
            return np.ones(shape, dtype=dtype)

    # Other types
    elif typ == str:
        return ""
    elif typ == int:
        return 1
    elif typ == float:
        return 1
    elif typ == Any:
        return None
    elif typ == PlantFunctionalTypeStrict:
        return PlantFunctionalType(name=f"default.{randint(1, 10000)}")
    elif len(inspect.signature(typ).parameters) > 0:
        return initialise_class(typ, shape)
    else:
        return typ()


manual_test_parameters = {
    # 'Bounds.lower': 0,
    # 'Bounds.interval_type': '[]',
    'Cohorts.__init__.dbh_values': np.zeros(3),
    'Cohorts.__init__.n_individuals': np.ones(3),
    'Cohorts.__init__.pft_names': np.array(['Tree1', 'Tree2', 'Tree3'], dtype=np.str_),
    'Flora.__init__.pfts': [PlantFunctionalType(name='Tree1'), PlantFunctionalType(name='Tree2'), PlantFunctionalType(name='Tree3')],
    'Canopy.__init__.fit_ppa': True,
}


def generate_args(name, method, shape):
    """Generate the arguments needed for a function. Requires type hinting.
    Numpy arrays are defined using the shape argument.
    """
    from typing import get_type_hints

    kwargs = {}
    sig = inspect.signature(method)
    for param_name, param in sig.parameters.items():
        method_arg_name = f"{name}.{param_name}"
        # Set manually defined values
        if method_arg_name in manual_test_parameters:
            kwargs[param_name] = manual_test_parameters[method_arg_name]
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
                raise Exception(f"Missing annotation for {method_arg_name}")
            # Get the contexts of the pyrealm classes to pass to get_type_hints
            globalns = {}
            for module in get_package_modules(pyrealm):
                globalns.update(vars(module))
            # Resolve any string annotations
            typ = get_type_hints(method, globalns=globalns).get(
                param_name, param.annotation
            )
            kwargs[param_name] = initialise_type_default(typ, shape)
    return kwargs


def initialise_class(cls, shape):
    init_name = f"{cls.__name__}.__init__"
    print("Initialising:", init_name)
    args = generate_args(init_name, cls.__init__, shape)
    return cls(**args)


def is_equal(val1, val2):
    if isinstance(val1, np.ndarray):
        return np.all(val1 == val2)

    elif isinstance(val1, tuple) and isinstance(val2, tuple):
        if len(val1) != len(val2):
            return False
        return all(is_equal(v1, v2) for v1, v2 in zip(val1, val2))

    elif hasattr(val1, "__dict__") and hasattr(val2, "__dict__"):
        return compare_instances(val1, val2)

    else:
        return val1 == val2


def compare_instances(instance1, instance2):
    dict1 = instance1.__dict__
    dict2 = instance2.__dict__
    for key in dict1:
        if not is_equal(dict1[key], dict2[key]):
            return False
    return True


shape = (3, 1, 1)
shape_full = (3, 2, 2)

for mod in get_package_modules(pyrealm):
    for name, method, cls in get_module_callables(mod):
        if not has_array_input(method):
            continue

        print(name)

        # Generate the arguments for the method
        args = generate_args(name, method, shape)
        args_full = generate_args(name, method, shape_full)

        if is_instance_method(cls, method.__name__):
            print("CM", name)
            # First initialise class and get bound methods
            instance1 = initialise_class(cls, shape)
            instance2 = initialise_class(cls, shape_full)
            method1 = getattr(instance1, method.__name__)
            method2 = getattr(instance2, method.__name__)
            # Run the method
            result = method1(**args)
            result_full = method2(**args_full)
            # Compare results
            is_correct = is_equal(result, result_full) and compare_instances(
                instance1, instance2
            )
            print("CM", name, is_correct)

        else:
            print("FN", name)
            # Run the method
            result = method(**args)
            result_full = method(**args_full)
            # Compare results
            is_correct = is_equal(result, result_full)
            print("FN", name, is_correct)

        print()
        # # Compare the results to see if they are identical
        # if (not is_correct):
        # raise Error(f'Results do not match in {name}')
