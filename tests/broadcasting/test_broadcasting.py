"""This module contains the tests to check broadcastable array inputs.

This ensures that the outputs/attributes of any functions/methods are unchanged when
broadcastable array inputs are used in place of the full size arrays.

To resolve failing tests there are a few options to use in utils.py:
    - `skip_methods`: For irrelevant methods (e.g. 1D inputs only).
    - `ignore_outputs`: To ignore comparing specific outputs or class attributes.
    - `defined_method_args`: To manually define specific arguments for methods.
    - `additional_init_methods`: For classes needing to call additional methods.
"""

import warnings
from collections.abc import Callable

import pytest
from utils import (
    Context,
    compare_instances,
    comparison_string,
    generate_args,
    get_method_list,
    initialise_class,
    is_equal,
)

from pyrealm.core.experimental import ExperimentalFeatureWarning

warnings.filterwarnings("ignore", category=ExperimentalFeatureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


shape_full: list[tuple[int, ...]]
shape_full = [(3, 2, 2)]
shapes_list = [
    [(3, 2, 2), (1, 2, 2), (3, 1, 1), (1, 1, 1)],
    [(1, 2, 2), (3, 2, 2)],
    [(3, 1, 1), (1, 2, 2), (3, 2, 2)],
    [(1, 1, 1)],
]
method_list = get_method_list()


@pytest.mark.parametrize("shapes", shapes_list)
@pytest.mark.parametrize("method_info", method_list, ids=[m[0] for m in method_list])
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
    ctx_full = Context(name, shape_full)
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


if __name__ == "__main__":
    for method_info in method_list:
        for shapes in shapes_list:
            test_array_input_broadcasting(method_info, shapes)
