"""This module contains the tests to check using xarray inputs.

This ensures that the outputs/attributes of any functions/methods are unchanged when
using xarray.DataArrays instead of numpy arrays.
"""

from collections.abc import Callable

import numpy as np
import pytest
import xarray as xr

from tests.broadcasting.utils import (
    Context,
    compare_instances,
    comparison_string,
    generate_args,
    get_method_list,
    initialise_class,
    is_equal,
)

SHAPES_LIST: list[list[tuple[int, ...]]] = [
    [(3, 2, 2), (1, 2, 2), (3, 1, 1), (1, 1, 1)],
    [(3, 2, 2), (2, 2), (3,), (1,)],
]
DIMS_LIST = [
    [],
    [("a", "b", "c"), ("b", "c"), ("a"), ("b")],
]
METHOD_LIST = get_method_list("ArrayType")


@pytest.mark.broadcasting
@pytest.mark.parametrize("shapes,dims_list", zip(SHAPES_LIST, DIMS_LIST))
@pytest.mark.parametrize("method_info", METHOD_LIST, ids=[m[0] for m in METHOD_LIST])
@pytest.mark.filterwarnings("ignore::ExperimentalFeatureWarning")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_xarray_input(
    method_info: tuple[str, Callable, type | None],
    shapes: list[tuple[int, ...]],
    dims_list: list[tuple[str, ...]],
):
    """Test to check that array input type doesn't affect the results for all functions.

    Each method / function is run twice. Once with all array inputs as xarray DataArrays
    and another with numpy arrays. Then compare the outputs (and all class attributes
    for class methods).
    """
    name, method, cls = method_info

    # Define the Contexts with the options for creating the arrays
    ctx_xr = Context(name, shapes, array_type=xr.DataArray, array_dims_list=dims_list)
    ctx_np = Context(name, SHAPES_LIST[0], array_type=np.array)

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
    if not is_equal(result_xr, result_np):
        result_comparison = comparison_string(result_xr, result_np)
        raise ValueError(f"Results do not match in {name} ({result_comparison})")
