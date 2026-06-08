"""Thought experiment on driving broadcast testing from decorating the codebase.

.. code:: python

    @_array_testing(_ArrayTesting())
    def func_one_test(a: int = 1) -> int:

        return a

    @_array_testing(_ArrayTesting(), _ArrayTesting())
    def func_two_test(a: int = 1) -> int:

        return a

    @_array_testing(_ArrayTesting())
    class Klass:

        def __init__(self, a: int = 1) -> None:
            self.a = a

        @_array_testing(_ArrayTesting())
        def func(self) -> int:
            return self.a

"""

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

import numpy as np

ARRAY_TESTING_REGISTRY: list = []


@dataclass
class _ArrayTesting:
    """Dataclass for array testing metadata."""

    array_args: tuple[tuple[str, float | np.generic], ...] = tuple()
    """Identifies the array inputs to a callable and a fill value"""
    suffix: str = ""
    """An optional suffix to distinguish subtests for the same callable."""
    test_attributes: tuple[str, ...] = tuple()
    """For class tests, which attributes should be tested for equality"""

    test_name: str = field(init=False)
    """Attribute to record a unique test name for the instance, set externally."""


def _array_testing(*args: _ArrayTesting) -> Callable:
    """Decorator to add a callable to the array testing suite.

    The decorator registers the callable as part of the testing suite, so developers can
    opt in to array testing, and attaches array testing metadata to callables for use in
    running the tests.
    """

    def attr_decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args: tuple[Any], **kwargs: Any) -> Callable:
            return fn(*args, **kwargs)

        # Populate the test_name for each _ArrayTesting instance provided in the
        # decorator arguments.
        for a in args:
            a.test_name = f"{fn.__module__}.{fn.__qualname__}"
            if a.suffix:
                a.test_name += f"_{a.suffix}"

        # Store array testing attributes on the code object? This creates a list of
        # _ArrayTesting instance stored inside the method metadata. Advantage of storing
        # on the object could be that the information can be used to generate class
        # instances on the fly for testing methods.
        setattr(wrapper, "_array_testing", args)

        # Register that the code object should be subjected to array testing
        for test_instance in args:
            ARRAY_TESTING_REGISTRY.append((fn, test_instance))

        return wrapper

    return attr_decorator
