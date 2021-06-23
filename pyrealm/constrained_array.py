import numpy as np
import warnings


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

    def __new__(cls, input_array: np.ndarray, lower=-np.infty, upper=np.infty):

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
        obj.constrained = True
        obj.constraints = [lower, upper]
        obj.n_constrained = obj.mask.sum() - n_masked

        return obj


def constraint_factory(label: str, lower=-np.infty, upper=np.infty):
    r"""This is a function factory used to create ConstrainedArrays
    with different limits. The generated functions issue a warning when the
    constraints are enforced using the provided label to aid reporting.

    Parameters:

        label: A string to be used when reporting constraints being enforced.
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
    """

    def action(arr: np.ndarray):

        arr = ConstrainedArray(arr, lower=lower, upper=upper)

        if arr.n_constrained:
            warnings.warn(f'{arr.n_constrained} {label} values outside '
                          f'[{lower}, {upper}] masked.',
                          category=RuntimeWarning)
        return arr

    return action
