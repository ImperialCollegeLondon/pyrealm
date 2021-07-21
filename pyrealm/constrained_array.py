import numpy as np
import warnings
from typing import Union, Callable
from numbers import Number

# TODO - include interval type as '[]', '()', '[)', '[i', etc.
#        using that str to set the bounding ufunc

# TODO - at the moment, functions using ConstrainedArray only check
#        that they are getting a CA, not that it has the _right_ constraints
#        Being paranoid, a user could create CAs and pass them in incorrectly
#        by getting the argument order wrong.
#        Maybe add an id attribute and also a check_constraint(arr, factory)
#        helper function.
#        This can still be defeated but a user would have to deliberately set
#        an incorrect ID.


class ConstrainedArray(np.ma.core.MaskedArray):
    r"""Array with constrained values

    This is a subclass of np.ndarray that constrains the range of permitted
    values allowed in the array by applying a mask to the original data.
    This is a general structure to make it easier to apply a constraint and
    detect when an input has already been constrained. It is based on code
    from: https://numpy.org/doc/stable/user/basics.subclassing.html

    Parameters:

        input_array: An np.ndarray object
        lower: The value of the lower constraint
        upper: The value of the upper constraint
        label: A string giving a descriptive label of the constrained contents

    Returns:

        A ConstrainedArray object, inheriting from np.ma.core.MaskedArray
        containing the original data but with values outside the constraints
        masked.

    Examples:

        >>> vals = np.array([-15, 20, 30, 124])
        >>> vals.sum()
        159
        >>> vals_c = ConstrainedArray(vals, 0, 100)
        >>> vals_c.sum()
        50
    """

    def __new__(cls, input_array: np.ndarray,
                lower: Number = -np.infty,
                upper: Number = np.infty,
                label: str = None):

        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)

        # Count the number of already masked cells
        if np.ma.is_masked(obj):
            n_masked = obj.mask.sum()
        else:
            n_masked = 0

        # Mask using the constraints as limits
        obj = np.ma.masked_outside(obj, v1=lower, v2=upper)

        # Record some details of the constraint
        obj.limits = [lower, upper]
        obj.n_constrained = obj.mask.sum() - n_masked
        obj.label = label

        return obj


def constraint_factory(label: str,
                       lower: Number = -np.infty,
                       upper: Number = np.infty):
    r"""This is a function factory used to create functions to generate
    ConstrainedArrays with preset limits and labels. The generated functions
    issue a warning when the constraints are enforced using the provided label
    to aid reporting.

    The resulting function also acts a validator. If the input is already
    a ConstrainedArray, the label and limits are compared to check that the


    Parameters:

        label: A string giving a descriptive label of the constrained contents
        lower: The lower constraint to enforce
        upper: The upper constraint to enforce

    Examples:

        >>> temp_constraint = constraint_factory('temperature', 0, 100)
        >>> vals = np.array([-15, 20, 30, 124])
        >>> vals.sum()
        159
        >>> vals_c = temp_constraint(vals)
        >>> vals_c.sum()
        50
        >>> diff = ConstrainedArray(vals, -50, 120, 'foo')
        >>> diff = temp_constraint(diff)
        Traceback (most recent call last):
        ...
        RuntimeError: Existing input constraints do not match checking constraints
        """

    def action(arr: np.ndarray):

        if isinstance(arr, ConstrainedArray):
            if arr.limits != [lower, upper] or arr.label != label:
                raise RuntimeError('Existing input constraints do not match checking constraints')
            else:
                return arr

        arr = ConstrainedArray(arr, lower=lower, upper=upper, label=label)

        if arr.n_constrained:
            warnings.warn(f'{arr.n_constrained} {label} values outside '
                          f'[{lower}, {upper}] masked.',
                          category=RuntimeWarning)
        return arr

    # Annotate the function with the limits and label to facilitate checking.
    action.limits = [lower, upper]
    action.label = label

    # Add a docstring
    action.__doc__ = "Test"

    return action


