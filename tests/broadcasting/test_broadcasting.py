"""This module contains the tests to check broadcastable array inputs.

This ensures that the outputs/attributes of any functions/methods are unchanged when
broadcastable array inputs are used in place of the full size arrays.

To resolve failing tests there are a few options to use in utils.py:
    - `SKIP_METHODS`: For irrelevant methods (e.g. 1D inputs only).
    - `IGNORE_OUTPUTS`: To ignore comparing specific outputs or class attributes.
    - `defined_method_args`: To manually define specific arguments for methods.
    - `ADDITIONAL_INIT_METHODS`: For classes needing to call additional methods.
"""

from collections.abc import Callable

import pytest

from tests.broadcasting.utils import (
    Context,
    compare_instances,
    comparison_string,
    generate_args,
    get_method_list,
    initialise_class,
    is_equal,
)

SHAPE_FULL: list[tuple[int, ...]]
SHAPE_FULL = [(3, 2, 2)]
SHAPES_LIST = [
    [(3, 2, 2), (1, 2, 2), (3, 1, 1), (1, 1, 1)],
    [(1, 2, 2), (3, 2, 2)],
    [(3, 1, 1), (1, 2, 2), (3, 2, 2)],
    [(1, 1, 1)],
]
METHOD_LIST = get_method_list("numpy")


@pytest.mark.broadcasting
@pytest.mark.parametrize("shapes", SHAPES_LIST)
@pytest.mark.parametrize("method_info", METHOD_LIST, ids=[m[0] for m in METHOD_LIST])
@pytest.mark.filterwarnings("ignore::ExperimentalFeatureWarning")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_array_input_broadcasting(
    method_info: tuple[str, Callable, type | None],
    shapes: list[tuple[int, ...]],
):
    """Test to run all module callables to check if broadcasting affects the results.

    Each method / function is run twice. Once with all array inputs in their full
    broadcasted shape, and another with equivalent, broadcastable inputs. Then compare
    the outputs (and all class attributes for class methods). Raises a ValueError if
    incorrect.
    """
    name, method, cls = method_info

    # Generate the arguments for the function / method
    ctx = Context(name, shapes)
    ctx_full = Context(name, SHAPE_FULL)

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
    if not is_equal(result, result_full, broadcast=True):
        result_comparison = comparison_string(result, result_full)
        raise ValueError(f"Results do not match in {name} ({result_comparison})")
