"""Draft code for the subdaily PModel."""  # noqa: D205, D415

import bottleneck as bn  # type: ignore
import numpy as np
from numpy.typing import NDArray

from pyrealm.pmodel.functions import calc_ftemp_arrh


def memory_effect(values: NDArray, alpha: float = 0.067) -> NDArray:
    r"""Apply a memory effect to a time series.

    Vcmax and Jmax do not converge instantaneously to acclimated optimal values. This
    function estimates how the actual Vcmax and Jmax track a time series of calculated
    optimal values assuming instant acclimation.

    The estimation uses the paramater `alpha` (:math:`\alpha`) to control the speed of
    convergence of the estimated values (:math:`E`) to the calculated optimal values
    (:math:`O`):

    ::math

        E_{t} = E_{t-1}(1 - \alpha) + O_{t} \alpha

    For :math:`t_{0}`, the first value in the optimal values is used so :math:`E_{0} =
    O_{0}`.

    Args
        values: An equally spaced time series of values
        alpha: The relative weight applied to the most recent observation

    Returns
        An np.ndarray of the same length as `values` with the memory effect applied.
    """

    # TODO - NA handling
    # TODO - think about filters here - I'm sure this is a filter which
    #        could generalise to longer memory windows.
    # TODO - need a version that handles time slices for low memory looping
    #        over arrays.

    memory_values = np.empty_like(values, dtype=np.float32)
    memory_values[0] = values[0]

    for idx in range(1, len(memory_values)):
        memory_values[idx] = memory_values[idx - 1] * (1 - alpha) + values[idx] * alpha

    return memory_values


def interpolate_rates_forward(
    tk: NDArray, ha: float, values: NDArray, values_idx: NDArray
) -> NDArray:
    """Interpolate Jmax and Vcmax forward in time.

    This is a specialised interpolation function used for Jmax and Vcmax. Given a time
    series of temperatures in Kelvin (`tk`) and a set of Jmax25 or Vcmax25 values
    observed at indices (`values_idx`) along that time series, this pushes those values
    along the time series and then rescales to the observed temperatures.

    The effect is that the plant 'sets' its response at a given point of the day and
    then maintains that same behaviour until a similar reference time the following day.

    Note that the beginning of the sequence will be filled with np.nan values unless
    values_idx[0] = 0.

    Arguments:
        tk: A time series of temperature values (Kelvin).
        ha: An Arrhenius constant.
        values: An array of rates at standard temperature predicted at points along tk.
        values_idx: The indices of tk at which values are predicted.
    """

    v = np.empty_like(tk)
    v[:] = np.nan

    v[values_idx] = values
    v = bn.push(v)

    return v * calc_ftemp_arrh(tk=tk, ha=ha)
