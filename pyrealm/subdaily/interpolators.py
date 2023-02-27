"""Draft code for subdaily interpolators."""  # noqa: D205, D415

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d  # type: ignore


class TemporalInterpolator:
    """Create a temporal interpolation of a variable.

    Instances of this class set a mapping from a coarser set of datetimes to a
    finer set of datetimes. Creating an instance sets up the interpolation time
    scales, and the instance can be called directly to interpolate a specific
    set of values.

    Interpolation uses :func:`scipy.interpolate.interp1d`. This provides a range
    of interpolation kinds available, with 'linear' probably the most
    appropriate, but this class adds a method `daily_constant`, which is
    basically the existing `previous` kind but offset so that a single value in
    a day is used for _all_ interpolated values in the day, including
    extrapolation backwards and forwards to midnight.

    Both inputs must be provided as arrays of type np.datetime64. This type has
    a range of subtypes with varying precision (e.g. 'datetime64[D]' for days
    and 'datetime64[s]' for seconds): the two input arrays _must_ use the same
    temporal resolution type.

    Args:
        input_datetimes: A numpy np.datetime64 array giving the datetimes of the
            observations
        interpolation_datetimes: A numpy np.datetime64 array giving the points
            at which to interpolate the observations.
    """

    def __init__(
        self,
        input_datetimes: NDArray[np.datetime64],
        interpolation_datetimes: NDArray[np.datetime64],
        method: str = "daily_constant",
    ) -> None:
        # This might be better as a straightforward function - there isn't a
        # huge amount of setup in __init__, so not saving a lot of processing by
        # saving that setup in class attributes for re-use.

        # TODO - this might be much more efficient with time as the _last_ axis
        # due to the organisation of contiguous memory in arrays.

        # There are some fussy things here with what is acceptable input to
        # interp1d: although the interpolation function can be created with
        # datetime.datetime or np.datetime64 values, the interpolation call
        # _must_ use float inputs (see https://github.com/scipy/scipy/issues/11093),
        # so internally this class uses floats from the inputs for
        # interpolation. Because the same datetime with different np.datetime64
        # subtypes gives different float values, the two arrays must use the
        # same subtype.

        # Inputs must be datetime64 arrays and have the same np.datetime64
        # subtype
        if not (
            np.issubdtype(input_datetimes.dtype, np.datetime64)
            and np.issubdtype(interpolation_datetimes.dtype, np.datetime64)
        ):
            raise TypeError("Interpolation times must be np.datetime64 arrays")

        if input_datetimes.dtype != interpolation_datetimes.dtype:
            raise TypeError("Inputs must use the same np.datetime64 precision subtype")

        self._method = method
        self._interpolation_x = interpolation_datetimes.astype("float")

        if method == "daily_constant":
            # This approach repeats the daily value for all subdaily times _on that
            # day_ and so should extrapolate beyond the last ref_datetime to the
            # end of that day and before the first ref_datetime to the beginning
            # of that day

            # Round observation times down to midnight on the day and append midnight
            # on the following day

            midnight = input_datetimes.astype("datetime64[D]").astype(
                input_datetimes.dtype
            )
            midnight = np.append(midnight, midnight[-1] + np.timedelta64(1, "D"))
            self._input_x = np.array(midnight).astype(float)

        else:
            self._input_x = input_datetimes.astype(float)

    def __call__(self, values: NDArray) -> NDArray:
        """Apply temporal interpolation to a variable.

        Calling an instance of :class:`~pyrealm.utilties.TemporalInterpolator`
        with a variable applies the temporal interpolation set in the instance
        to the inputs and returns an interpolated variable, using the method set
        when the instance was created.

        Args:
            values: A numpy array of numeric values, of the same length as the
            `input_datetimes` used to create the instance.

        Returns:
            A numpy array of values interpolated to the timepoints in the
            `interpolation_datetimes` values used to create the instance.
        """

        if self._method == "daily_constant":
            # Append the last value to match the appended day
            values = np.array(values)
            values = np.append(values, values[-1])
            method = "previous"
        else:
            method = self._method

        # Check the first axis of the values has the same length as the input
        # datetimes.
        if len(self._input_x) != values.shape[0]:
            raise ValueError(
                "The first axis of values does not match the length of input_datetimes"
            )

        interp_fun = interp1d(
            self._input_x, values, axis=0, kind=method, bounds_error=False
        )

        return interp_fun(self._interpolation_x)
