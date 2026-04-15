"""This module contains the tests to check using xarray inputs.

This ensures that the outputs/attributes of any functions/methods are unchanged when
using xarray.DataArrays instead of numpy arrays.
"""

import itertools
import random
from collections.abc import Callable
from functools import partial

import pytest

from tests.array_inputs.utils import (
    Context,
    assert_is_equal,
    compare_instances,
    generate_args,
    get_method_list,
    initialise_class,
)

METHOD_LIST = get_method_list("ArrayType")


# List of functions that must match a previously defined shape
DEPENDENT_LIST: list[str] = [
    # pmodel
    "PModel.apply_gpp_penalty_factor",
    "SubdailyPModel.apply_gpp_penalty_factor",
    "CarbonIsotopes",
    "C3C4Competition.estimate_isotopic_discrimination",
    "OptimalChiPrentice14.estimate_chi",
    "OptimalChiPrentice14RootzoneStress.estimate_chi",
    "OptimalChiC4.estimate_chi",
    "OptimalChiC4RootzoneStress.estimate_chi",
    "OptimalChiLavergne20C3.estimate_chi",
    "OptimalChiLavergne20C4.estimate_chi",
    "OptimalChiC4NoGamma.estimate_chi",
    "OptimalChiC4NoGammaRootzoneStress.estimate_chi",
    # splash
    "SplashModel.estimate_initial_soil_moisture",
    "SplashModel.estimate_daily_water_balance",
    "SplashModel.calculate_soil_moisture",
    "DailyEvapFluxes",
    "DailyEvapFluxes.estimate_aet",
    # phenology
    "FaparLimitation.from_pmodel",
]

# Check all dependent functions are used
unused: set[str] = set(DEPENDENT_LIST) - {m[0] for m in METHOD_LIST}
assert not unused, f"Dependent functions not in METHOD_LIST: {unused}"


def shapes_xarray(
    i_shapes: int, i_array: int, n_array: int, name: str
) -> dict[str, int]:
    """Function to generate the shapes for xarray arguments.

    The first run uses the same shapes to check basic conversion. The later ones remove
    and rearrange dimensions under the constraints:
    - The combination of all arguments must give the full shape and set of dimensions
      (unless it is in DEPENDENT_LIST).
    - The order of initial dimensions must be "a", "b", "c".
    """
    full_shape = {"a": 2, "b": 3, "c": 4}

    if i_shapes == 0:
        return full_shape

    elif i_shapes == 1 and name in DEPENDENT_LIST:
        return {"b": 3}

    shape = full_shape.copy()
    dims = list(shape.keys())

    # Randomly choose which array will have the first of each dimension
    # Constraining initial order of a, b, c
    first_idx = [0]
    random.seed((i_shapes + 1) * n_array)  # Same seed for all arrays
    for i_dim in range(1, len(dims)):
        first_idx.append(random.randrange(first_idx[i_dim - 1], n_array))
    # No constraint if function in DEPENDENT_LIST
    if name in DEPENDENT_LIST:
        first_idx = [-1 for _ in range(len(dims))]
    first_occurrence = {dim: first_idx[i] == i_array for i, dim in enumerate(dims)}

    # Randomly drop dimensions
    random.seed((i_shapes + 1) * n_array + i_array)  # Different seed for each array
    for i_dim, dim in enumerate(dims):
        # If the array is before the first index, delete dim
        if i_array < first_idx[i_dim]:
            del shape[dim]

        # If the array is equal to the first index, keep dim
        elif first_occurrence[dim]:
            continue

        # Otherwise, randomly choose whether to delete dim
        elif random.choice([True, False]):
            del shape[dim]

    # If empty restore a dimension
    if len(shape) == 0:
        eligible_dims = [
            dim
            for i_dim, dim in enumerate(full_shape.items())
            if i_array >= first_idx[i_dim]
        ]
        shape = dict([random.choice(eligible_dims)])

    # Update remaining dimensions
    dims = list(shape.keys())

    # Reorder dimensions while constraining the initial order of a, b, c
    # Don't reorder if this and the following dimension are both the first occurrence
    fixed = {dim: False for dim in dims}
    for dim, next_dim in itertools.pairwise(dims):
        if first_occurrence[dim] and first_occurrence[next_dim]:
            fixed[dim] = True
            fixed[next_dim] = True
    # Reorder the non-fixed dims and then re-insert the fixed dims
    reordered_dims = [dim for dim in dims if not fixed[dim]]
    random.shuffle(reordered_dims)
    for i_dim, dim in enumerate(dims):
        if fixed[dim]:
            reordered_dims.insert(i_dim, dim)
    # Return the reordered shape
    return {dim: shape[dim] for dim in reordered_dims}


def shapes_numpy(
    i_shapes: int, i_array: int, n_array: int, name: str
) -> dict[str, int]:
    """Function to get the full set of dimensions. Missing dimensions -> size 1."""
    shapes_xr = shapes_xarray(i_shapes, i_array, n_array, name)
    return {dim: shapes_xr.get(dim, 1) for dim in ["a", "b", "c"]}


@pytest.mark.array_inputs
@pytest.mark.parametrize("i_shapes", range(3))
@pytest.mark.parametrize("method_info", METHOD_LIST, ids=[m[0] for m in METHOD_LIST])
@pytest.mark.filterwarnings("ignore::ExperimentalFeatureWarning")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_xarray_input(method_info: tuple[str, Callable, type | None], i_shapes: int):
    """Test to check that array input type doesn't affect the results for all functions.

    Each method / function is run twice. Once with all array inputs as xarray DataArrays
    and another with numpy arrays. Then compare the outputs (and all class attributes
    for class methods).
    """
    name, method, cls = method_info

    # Define the Contexts with the options for creating the arrays
    ctx_xr = Context(name, partial(shapes_xarray, i_shapes), array_type="xarray")
    ctx_np = Context(name, partial(shapes_numpy, i_shapes), array_type="numpy")

    # If a class method (initialises class and compares attributes)
    if cls is not None:
        # First initialise class and get bound methods - the initialise class function
        # calls generate_args() internally for the __init__ method
        instance_xr = initialise_class(cls, ctx_xr)
        instance_np = initialise_class(cls, ctx_np)

        if method.__name__ == "__init__":
            result_xr = None
            result_np = None
        else:
            # Get the method attribute from the class
            method_xr = getattr(instance_xr, method.__name__)
            method_np = getattr(instance_np, method.__name__)

            # Generate the arguments to run the method and run it
            result_xr = method_xr(**generate_args(method, ctx_xr))
            result_np = method_np(**generate_args(method, ctx_np))

        # Fail if attributes not equal
        compare_instances(instance_xr, instance_np)

    # If a function / static method
    else:
        # Run the method
        result_xr = method(**generate_args(method, ctx_xr))
        result_np = method(**generate_args(method, ctx_np))

    # Fail if function outputs not equal
    assert_is_equal(result_xr, result_np, raise_msg=f"Results do not match in {name}")


if __name__ == "__main__":
    # Debugging: check the output of the array shapes
    for n_array in range(1, 6):
        print(f"\n{n_array} array argument(s):")
        for i_shapes in range(3):
            print(f"Run {i_shapes}:")
            for i_array in range(n_array):
                print(shapes_xarray(i_shapes, i_array, n_array, ""))
