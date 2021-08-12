import numpy as np
import warnings
from numbers import Number


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
        interval_type: The interval type of the constraint ('[]', '()', '[)', '(]')
        label: A string giving a descriptive label of the constrained contents

    Returns:

        A ConstrainedArray object, inheriting from np.ma.core.MaskedArray
        containing the original data but with values outside the constraints
        masked.

    Examples:

        >>> vals = np.array([-15, 20, 30, 124])
        >>> vals.sum()
        159
        >>> vals_c = ConstrainedArray(vals, 0, 100, label='temperature')
        >>> vals_c.sum()
        50
        >>> vals_c = ConstrainedArray(vals, 0, 124, interval_type='[]', label='temperature')
        >>> vals_c.sum()
        174
        >>> vals_c = ConstrainedArray(vals, 0, 124, interval_type='[)', label='temperature')
        >>> vals_c.sum()
        50
    """

    def __new__(cls, input_array: np.ndarray,
                lower: Number = -np.infty,
                upper: Number = np.infty,
                interval_type: str = '[]',
                label: str = ''):

        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)

        # Count the number of already masked cells if any
        if np.ma.is_masked(obj):
            n_masked = obj.mask.sum()
        else:
            n_masked = 0

        # Implement the interval type
        if interval_type not in ['[]', '()', '[)', '(]']:
            raise RuntimeWarning(f'Unknown interval type: {interval_type}')

        if interval_type[0] == '[':
            lower_func = np.less
        else:
            lower_func = np.less_equal

        if interval_type[1] == ']':
            upper_func = np.greater
        else:
            upper_func = np.greater_equal

        # Mask using the constraints as limits
        mask = np.logical_or(lower_func(obj, lower), upper_func(obj, upper))
        obj = np.ma.masked_where(mask, obj)

        # Record some details of the constraint
        obj.lower = lower
        obj.upper = upper
        obj.interval_type = interval_type
        obj.n_constrained = obj.mask.sum() - n_masked
        obj.label = label
        obj.interval_notation = f"{interval_type[0]}{lower}, {upper}{interval_type[1]}"

        if obj.n_constrained:
            warnings.warn(f'{obj.n_constrained} {obj.label} values outside '
                          f'{obj.interval_notation} masked.',
                          category=RuntimeWarning)
        return obj


class ConstraintFactory:
    r"""This is a function factory used to create functions to generate
    ConstrainedArrays with preset limits and labels. The generated functions
    issue a warning when the constraints are enforced using the provided label
    to aid reporting.

    The resulting function also acts a validator. If the input is already
    a ConstrainedArray, then the function checks that the label and interval
    match the factory.

    Parameters:

        lower: The value of the lower constraint
        upper: The value of the upper constraint
        interval_type: The interval type of the constraint ('[]', '()', '[)', '(]')
        label: A string giving a descriptive label of the constrained contents

    Examples:

        >>> temp_constraint = ConstraintFactory(0, 100, label='temperature (°C)')
        >>> temp_constraint
        ConstraintFactory: temperature (°C) constrained to [0, 100]
        >>> vals = np.array([-15, 20, 30, 124])
        >>> vals.sum()
        159
        >>> vals_c = temp_constraint(vals)
        >>> vals_c.sum()
        50
        >>> diff = ConstrainedArray(vals, -50, 120, label='foo')
        >>> diff = temp_constraint(diff)
        Traceback (most recent call last):
        ...
        RuntimeError: Existing input constraints ([-50, 120], foo) do not match checking constraints ([0, 100], temperature (°C))
        """

    def __init__(self,
                 lower: Number = -np.infty,
                 upper: Number = np.infty,
                 interval_type: str = '[]',
                 label: str = ''):

        self.label = label
        self.lower = lower
        self.upper = upper
        self.interval_type = interval_type
        self.interval_notation = f"{interval_type[0]}{lower}, {upper}{interval_type[1]}"

    def __call__(self, arr: np.ndarray):

        if isinstance(arr, ConstrainedArray):
            if not (arr.lower == self.lower and arr.upper == self.upper and
                    arr.interval_type == self.interval_type and arr.label == self.label):
                raise RuntimeError(f'Existing input constraints ({arr.interval_notation}, {arr.label}) '
                                   f'do not match checking constraints ({self.interval_notation}, {self.label})')
            else:
                return arr

        arr = ConstrainedArray(arr, lower=self.lower, upper=self.upper,
                               interval_type=self.interval_type, label=self.label)

        return arr

    def __repr__(self):

        return f"ConstraintFactory: {self.label} constrained to {self.interval_notation}"



