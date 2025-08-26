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

from .utils import (
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
METHOD_LIST = get_method_list()


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
    args = generate_args(method, ctx)
    args_full = generate_args(method, ctx_full)

    # If a class method (initialises class and compares attributes)
    if cls is not None:
        # First initialise class and get bound methods
        instance1 = initialise_class(cls, ctx)
        instance2 = initialise_class(cls, ctx_full)
        method1 = getattr(instance1, method.__name__)
        method2 = getattr(instance2, method.__name__)
        # Run the method
        result = method1(**args)
        result_full = method2(**args_full)
        # Fail if attributes not equal
        compare_instances(instance1, instance2)

    # If a function / static method
    else:
        # Run the method
        result = method(**args)
        result_full = method(**args_full)

    # Fail if function outputs not equal
    if not is_equal(result, result_full):
        result_comparison = comparison_string(result, result_full)
        raise ValueError(f"Results do not match in {name} ({result_comparison})")
