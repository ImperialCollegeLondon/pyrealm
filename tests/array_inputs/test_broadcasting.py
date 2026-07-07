"""This module contains the tests to check broadcastable array inputs.

This ensures that the outputs/attributes of any functions/methods are unchanged when
broadcastable array inputs are used in place of the full size arrays.

To resolve failing tests there are a few options to use in utils.py:
    - `SKIP_METHODS`: For irrelevant methods (e.g. 1D inputs only).
    - `IGNORE_OUTPUTS`: To ignore comparing specific outputs or class attributes.
    - `defined_method_args`: To manually define specific arguments for methods.
    - `ADDITIONAL_INIT_METHODS`: For classes needing to call additional methods.
"""

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

METHOD_LIST = get_method_list("numpy")


def shapes_full(i_array: int, n_array: int, name: str) -> dict[str, int]:
    """Function to pass to Context to get the full shape."""
    return {"a": 2, "b": 3, "c": 4}


def shapes(i_shapes: int, i_array: int, n_array: int, name: str) -> dict[str, int]:
    """Function to pass to Context to generate the shapes for array arguments.

    Each dimension can be 1 or the full shape and the first uses (1, 1, 1) for all
    inputs.
    """

    if i_shapes == 0:
        return {"a": 1, "b": 1, "c": 1}

    shape = {"a": 2, "b": 3, "c": 4}
    random.seed((i_shapes + 1) * n_array + i_array)
    for dim in shape:
        shape[dim] = random.choice([1, shape[dim]])
    return shape


@pytest.skip(reason="Significant issues with maintenance.")
@pytest.mark.array_inputs
@pytest.mark.parametrize("i_shapes", range(3))
@pytest.mark.parametrize("method_info", METHOD_LIST, ids=[m[0] for m in METHOD_LIST])
@pytest.mark.filterwarnings("ignore::ExperimentalFeatureWarning")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_array_input_broadcasting(
    method_info: tuple[str, Callable, type | None],
    i_shapes: int,
):
    """Test to run all module callables to check if broadcasting affects the results.

    Each method / function is run twice. Once with all array inputs in their full
    broadcasted shape, and another with equivalent, broadcastable inputs. Then compare
    the outputs (and all class attributes for class methods). Raises a ValueError if
    incorrect.
    """
    name, method, cls = method_info

    # Generate the arguments for the function / method
    ctx = Context(name, partial(shapes, i_shapes))
    ctx_full = Context(name, shapes_full)

    # If a class method (initialises class and compares attributes)
    if cls is not None:
        # First initialise class and get bound methods - the initialise class function
        # calls generate_args() internally for the __init__ method
        instance1 = initialise_class(cls, ctx)
        instance2 = initialise_class(cls, ctx_full)

        if method.__name__ == "__init__":
            result = None
            result_full = None
        else:
            # Get the method attribute from the class
            method1 = getattr(instance1, method.__name__)
            method2 = getattr(instance2, method.__name__)

            # Generate the arguments to run the method and run it
            result = method1(**generate_args(method, ctx))
            result_full = method2(**generate_args(method, ctx_full))

        # Fail if attributes not equal
        compare_instances(instance1, instance2, broadcast=True)

    # If a function / static method
    else:
        # Run the method
        result = method(**generate_args(method, ctx))
        result_full = method(**generate_args(method, ctx_full))

    # Fail if function outputs not equal
    raise_msg = f"Results do not match in {name}"
    assert_is_equal(result, result_full, raise_msg, broadcast=True)


if __name__ == "__main__":
    # Debugging: check the output of the array shapes
    for n_array in range(1, 6):
        print(f"\n{n_array} array argument(s):")
        for i_shapes in range(3):
            print(f"Run {i_shapes}:")
            for i_array in range(n_array):
                print(shapes(i_shapes, i_array, n_array, ""))
