"""Draft of decorator based broadcast testing."""

import importlib
import pkgutil
from itertools import chain, combinations
from typing import Any

import numpy as np
import pytest
import xarray

from pyrealm.core._array_testing import _ArrayTesting


def populate_array_test_callables():
    """Imports all modules to populate the array testing registry."""
    import pyrealm
    from pyrealm.core._array_testing import ARRAY_TESTING_REGISTRY

    for _, modname, ispkg in pkgutil.walk_packages(
        pyrealm.__path__, prefix=pyrealm.__name__ + "."
    ):
        if not ispkg:
            importlib.import_module(modname)

    return ARRAY_TESTING_REGISTRY


# All callables with @_array_testing decoration
ALL_ARRAY_TEST = populate_array_test_callables()

# Reduced set of callables that require more than one array input, used to check
# broadcasting reduced shapes to match full shapes works as expected
BROADCAST_TEST = [(cl, info) for cl, info in ALL_ARRAY_TEST if len(info.array_args) > 1]


def powerset(items):
    """Generate the powerset of a list of items.

    Implementation adapted from itertools powerset recipe. Returns a list of lists of
    items, excluding the empty set.
    """

    # Generate powerset of dimension combinations, ignoring empty set
    dim_powerset = chain.from_iterable(
        list(combinations(items, r)) for r in range(1, len(items) + 1)
    )

    # Return powerset values as lists for indexing numpy arrays
    return [list(v) for v in dim_powerset]


def equality_testing(full: Any, alt: Any, call_info: _ArrayTesting):
    """Simple equality test for comparing outputs.

    Functions typically return an array or tuple of arrays and so can be directly
    tested, but classes are hard to test so a set of defined attributes are tested
    instead.
    """
    if call_info.test_attributes:
        for attr in call_info.test_attributes:
            assert np.allclose(getattr(full, attr), getattr(alt, attr))
    else:
        assert np.allclose(full, alt)


@pytest.fixture(scope="module")
def collapsed_axis_combinations():
    """Generates all combinations of collapsed axis for 3D test arrays."""

    return powerset(range(3))


@pytest.mark.parametrize(
    "to_broadcast_test",
    BROADCAST_TEST,
    ids=[info.test_name for _, info in BROADCAST_TEST],
)
def test_array_broadcasting(to_broadcast_test, collapsed_axis_combinations):
    """Test numpy array broadcasting.

    Runs tests on callables with multiple array inputs to check that reduced rank inputs
    broadcast correctly to the full shape of the inputs. For example, if the data has
    shape (15, 10, 5), then do inputs with shape (1, 10, 5), etc give the same result.

    TODO: currently each subtest uses the same collapsed axis combination for all
          arguments but one - could randomise? Don't like random in tests much but here
          it could be a better test. Don't want to test everything because that gives
          combinatorial hell.
    """

    # Get the callable and the test inputs
    callable_, call_info = to_broadcast_test

    # Generate the result with full shape inputs for all array arguments
    full = np.array((15, 10, 5))
    full_args = {
        attr_name: np.full(full, attr_value)
        for attr_name, attr_value in call_info.array_args
    }
    full_result = callable_(**full_args)

    # Now iterate over collapsed axis combinations
    for axis_combination in collapsed_axis_combinations:
        # Generate the reduced axis shape for this axis combination.
        reduced = full.copy()
        reduced[axis_combination] = 1

        # Compile a template argument list with reduced shape
        reduced_args_template = {
            attr_name: np.full(reduced, attr_value)
            for attr_name, attr_value in call_info.array_args
        }

        # Restore one array attribute at a time to full shape and check the broadcast
        # result works.
        for full_name, _ in call_info.array_args:
            reduced_args = reduced_args_template.copy()
            reduced_args[full_name] = full_args[full_name]
            reduced_result = callable_(**reduced_args)

            equality_testing(full=full_result, alt=reduced_result, call_info=call_info)


@pytest.mark.parametrize(
    "to_array_test",
    ALL_ARRAY_TEST,
    ids=[info.test_name for _, info in ALL_ARRAY_TEST],
)
def test_xarray_conversion(to_array_test):
    """Test xarray conversion.

    Tests that substituting xarray for numpy inputs does not change result.
    """

    # Get the callable and the test inputs
    callable_, call_info = to_array_test

    # Generate the result with full shape inputs for all array arguments, typing here to
    # avoid complaints below when xarray DataArrays assigned over numpy arrays.
    full = np.array((15, 10, 5))
    full_args: dict[str, Any] = {  # type: ignore[annotation-unchecked]
        attr_name: np.full(full, attr_value)
        for attr_name, attr_value in call_info.array_args
    }
    full_result = callable_(**full_args)

    # Now iterate over powerset combinations of array arguments
    # TODO: this might be overkill for callables with large numbers of array inputs -
    #       maybe constrain to pairs or triplets if runtime explodes?
    for args_to_convert in powerset(full_args):
        # Generate the converted inputs for this combination.
        converted_args = full_args.copy()

        # Convert specified arrays to xarray
        for arg in args_to_convert:
            converted_args[arg] = xarray.DataArray(full_args[arg])

        # Run the callable and test
        converted_result = callable_(**converted_args)

        equality_testing(full=full_result, alt=converted_result, call_info=call_info)


def test_xarray_reshaping():
    """Test xarray reshaping.

    TODO - test the xarray code that maps missing dims and reshapes to common
    ordering.
    """

    pass
