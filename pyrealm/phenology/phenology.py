"""Computing phenological time series of Leaf Area Index.

This module provides classes to implement different approaches to calculating daily
predictions of leaf area index (LAI) from daily time series of potential assimilation.
The :class:`Phenology` class acts as a wrapper around alternative implementations of
the calculations, which are implemented as derived subclasses of the
:class:`PhenologyMethodABC` abstract base class.

The methods all require values of annual maximum fAPAR, calculated using
:class:`~pyrealm.phenology.fapar_limitation.FaparLimitation` and then a time
series of daily assimilation, although different methods can also require additional
inputs. The methods then broadly work by calculating a time series of steady state LAI
as a function of daily assimilation and then applying a lag, capturing the speed with
which plants can put on leaf area, to generate a time series of realised LAI.

Values of daily assimilation are typically going to be taken from a P Model, so the
:meth:`Phenology.from_pmodel` method is provided to calculate daily assimilation from
a P Model input and then use this, along with a ``FaparLimitation`` instance, to
generate an LAI time series.

The :class:`Phenology` class and the :meth:`Phenology.from_pmodel` are designed to
work with inputs that can have multiple dimensions. The first axis is _always_ assumed
to represent a time series of daily observations of potential assimilation. If all the
arrays are one dimensional, then this is a time series for a single site; if they are
three dimensional then these are observations for a grid of sites.

Usually all array inputs will have the same shape. However, where methods use a
climatological measure of aridity, then this is very likely to be constant through time.
If so, the array should have a singleton first dimension to broadcast an observation per
site across observations. For example, with `(365, 3, 3)` data (one year of observations
for a 3x3 grid of sites), the aridity index could be provided as `(1,3,3)` to broadcast
the aridity index across all observations. It could also use a single scalar value to
use the same aridity index for all sites.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import lambertw  # type: ignore[import-untyped]

from pyrealm.core.experimental import warn_experimental
from pyrealm.core.utilities import (
    check_input_shapes,
    exponential_moving_average,
)
from pyrealm.phenology.fapar_limitation import FaparLimitation
from pyrealm.pmodel.pmodel import PModel, PModelABC, SubdailyPModel

PHENOLOGY_METHOD_CLASS_REGISTRY: dict[str, type[PhenologyMethodABC]] = {}
"""A registry for classes implementing different methods for estimating phenology.

Different implementations of the calculation of LAI phenology must all be subclasses of
the :class:`~pyrealm.phenology.phenology.PhenologyMethodABC` abstract base class.
This dictionary is used as a registry for defined subclasses and a method name
is used to retrieve a particular implementation from this registry. For example:

.. code:: python

    zhou_phenology_class = PHENOLOGY_METHOD_CLASS_REGISTRY['zhou']
"""


class PhenologyMethodABC(ABC):
    r"""Abstract base class for implementations of LAI phenology calculation."""

    __experimental__ = True

    method: str
    """A short method name used to identify the class in
    :data:`PHENOLOGY_METHOD_CLASS_REGISTRY`.
    """

    attrs: tuple[tuple[str, str], ...]
    """A tuple of attributes to be reported in the Phenology.summarize() output
    for the method."""

    requires: tuple[str, ...]
    """A tuple of any additional variables required to create a Phenology
    instance using the method."""

    @abstractmethod
    def __init__(
        self,
        fapar_limitation: FaparLimitation,
        daily_potential_assimilation: NDArray[np.floating],
        datetimes: NDArray[np.datetime64],
        year_index: NDArray[np.int_],
        **kwargs: Any,
    ) -> None:
        """Calculate steady state and realise leaf area index.

        Each subclass should use this method to provide the method specific
        functionality to calculate steady state and realise leaf area index.
        """

        self.steady_state_lai: NDArray[np.floating]
        self.realised_lai: NDArray[np.floating]

    @classmethod
    def __init_subclass__(cls, method: str) -> None:
        """Initialise a subclass deriving from this ABC."""

        cls.method = method
        PHENOLOGY_METHOD_CLASS_REGISTRY[cls.method] = cls


class PhenologyMethodZhou(PhenologyMethodABC, method="zhou"):
    r"""Calculation of phenology following :cite:`zhou:2025a`.

    This phenology method class implements the calculation of daily leaf area index from
    daily assimilation (:math:`A_0`, mol m-2 day) following the method of
    :cite:`zhou:2025a`. The method requires annual values of fAPAR limitation calculated
    following the method of :cite:`cai:2025a` (see
    :class:`~pyrealm.phenology.fapar_limitation.FaparLimitationMethodCai`).

    * The fractional allocation of GPP to leaf area index (:math:`m`) is calculated
      from the annual maximum LAI and fAPAR as:

        .. math::

            m = \frac{\sigma G L_{max}}{A_{0} f_{APAR_{max}}}

      Here, $G$ and $A_0$ are the annual estimates of growing season length and annual
      total potential GPP used to calculate the annual maximum fAPAR and LAI. The
      coefficient :math:`\sigma` is a factor that accounts for the speed with which
      seasonal canopy growth and senescence patterns affect allocation to LAI (see
      :attr:`~pyrealm.constants.phenology_const.PhenologyConstNew.cai_sigma`).

    * The daily allocation of GPP to leaf area index (:math:`\mu`) is calculated simply
      as :math:`\mu = m A_{d0}`, where :math:`A_{d0}` is total daily potential
      assimilation.

    * The steady state daily estimate of LAI is then calculated as:

        .. math::

            L_s = \min \left\{
                \mu + \left(\frac{1}{k}\right) W_0 \left[ -k \mu \exp (-k \mu)\right],
                \text{LAI}_{\text{max}}
            \right\},

      where :math:`k` is the light extinction coefficient, :math:`W_0` is the principal
      branch of the Lambert W function. The value is constrained at an upper bound using
      the estimated annual maximum LAI from the calculate of fAPAR limitation.

    * The realised daily LAI is estimated using an exponential weighted average with a
      weight :math:`alpha` reflecting the time taken for plants to produce leaf area.
      The value of alpha is set in the phenology constants used to calculate the initial
      annual fAPAR limitation
      (:attr:`~pyrealm.constants.phenology_const.PhenologyConstNew.zhou_alpha`)
    """

    __experimental__ = True

    attrs = (
        ("steady_state_lai", "-"),
        ("realised_lai", "-"),
    )

    requires = tuple()

    def __init__(
        self,
        fapar_limitation: FaparLimitation,
        daily_potential_assimilation: NDArray[np.floating],
        datetimes: NDArray[np.datetime64],
        year_index: NDArray[np.int_],
        **kwargs: Any,
    ) -> None:
        """Calculate leaf area indices following Zhou et al."""

        # Run the super class ___init__.
        super().__init__(
            fapar_limitation=fapar_limitation,
            daily_potential_assimilation=daily_potential_assimilation,
            datetimes=datetimes,
            year_index=year_index,
            **kwargs,
        )

        phenology_const = fapar_limitation.phenology_const

        # Calculate the fractional allocation of GPP to LAI (m)
        self.lai_to_gpp_ratio_m: NDArray[np.floating] = (
            phenology_const.cai_sigma
            * fapar_limitation.annual_growing_season_length
            * fapar_limitation.lai_max
        ) / (fapar_limitation.annual_total_potential_gpp * fapar_limitation.fapar_max)
        """The steady state ratio of leaf area index to potential GPP (:math:`m`)"""

        # Duplicate m ratio for each day and calculate daily mu value as m * daily molar
        # assimilation:
        mu = self.lai_to_gpp_ratio_m[year_index] * daily_potential_assimilation

        # Calculate the Lambert W0 value
        daily_lai = mu + (1 / phenology_const.k) * lambertw(
            -phenology_const.k * mu * np.exp(-phenology_const.k * mu), k=0
        )

        # Check that all imaginary parts are zero or np.nan
        if not np.all(np.logical_or(np.imag(daily_lai) == 0, np.isnan(daily_lai))):
            raise ValueError("Imaginary parts of Lambert W calculation are not zero")

        # Clip the real parts at zero
        daily_lai = np.clip(np.real(daily_lai), a_min=0, a_max=None)

        # Find the daily minimum of the lambert term and annual maximum LAI as the
        # steady state LAI
        steady_state_lai = np.minimum(daily_lai, fapar_limitation.lai_max[year_index])
        """The steady state leaf area index for each day."""

        # Calculate the lagged value
        realised_lai = exponential_moving_average(
            steady_state_lai, alpha=phenology_const.zhou_alpha
        )
        """The realised leaf area index for each day given the modelled lag in responses
            to changing assimilation."""

        self.steady_state_lai = steady_state_lai
        self.realised_lai = realised_lai


class PhenologyMethodZhu(PhenologyMethodABC, method="zhu"):
    r"""Calculation of phenology following :cite:`zhu:2026a`.

    This phenology method class implements the calculation of daily leaf area index from
    daily assimilation (:math:`A_0`, mol m-2 day) following the method of
    :cite:`zhu:2026a` The method requires annual values of fAPAR limitation calculated
    following the method of :cite:`zhu:2026a` (see
    :class:`~pyrealm.phenology.fapar_limitation.FaparLimitationMethodZhu`).

    * The fractional allocation of GPP to LAI (:math:`m`) is calculated as the estimated
      maximum LAI for a year divided by a quantile value (default 0.95) from the
      distribution of daily assimilation values for that year. The quantile is set when
      calculating maximum fAPAR via the
      :attr:`PhenologyConstNew.zhu_A0_quantile<pyrealm.constants.phenology_const.PhenologyConstNew.zhu_A0_quantile>`
      attribute.

    * The steady state daily estimate of LAI is then calculated simply as:

        .. math::

            L_s = m A_0.

    * The realised daily LAI is estimated using a weighted average over the preceding N
      days of estimates of steady state LAI. The value of N varies with the
      climatological estimate of the AET/PET ratio for a site, which must be provided to
      use this method.

      For the first N observations, this weighted average is calculated over fewer
      observations - for the first observation it is simply the observed value. In order
      to avoid this, the method includes an optional spin up that can be used to
      condition the first values of preceding observation. The conditioning data is
      taken from the first year of the time series under the assumption that this will
      be a good approximation for the preceding conditions.

      .. TODO:

        I _think_ that we will get identical results by prepending the N observations
        from the end of the first year as we get from prepending the whole first year.
        If so then we can simplify this method to simply saying condition or not and
        then the length we need is simply the lag length N. We might be able to do the
        same for the Zhou method and make this general - but the exponential weighted
        average has a longer 'memory' than the simple inverse of alpha.
    """

    __experimental__ = True

    attrs = (
        ("steady_state_lai", "-"),
        ("realised_lai", "-"),
    )

    requires = ("aet_pet_ratio",)

    def __init__(
        self,
        fapar_limitation: FaparLimitation,
        daily_potential_assimilation: NDArray[np.floating],
        datetimes: NDArray[np.datetime64],
        year_index: NDArray[np.int_],
        aet_pet_ratio: NDArray[np.floating],
        spinup_length: int,
        **kwargs: Any,
    ) -> None:
        """Calculate leaf area indices following Zhou et al."""

        # Run the super class ___init__.
        super().__init__(
            fapar_limitation=fapar_limitation,
            daily_potential_assimilation=daily_potential_assimilation,
            datetimes=datetimes,
            year_index=year_index,
            **kwargs,
        )

        if aet_pet_ratio.shape != daily_potential_assimilation[[0]].shape:
            raise ValueError(
                "The 'aet_pet_ratio' must be an array providing one value per site."
            )

        # Calculate the steady state ratio of leaf area index to potential GPP
        phenology_const = fapar_limitation.phenology_const

        # Split the daily A0 up by year, calculate annual quantiles of daily
        # assimilation and then restack into an array, which should have the same shape
        # as the values in the fapar limitation.
        annual_A0_arrays = np.split(
            daily_potential_assimilation, np.where(np.diff(year_index))[0], axis=0
        )
        daily_A0_quantiles = np.stack(
            [
                np.nanquantile(ann_data, q=phenology_const.zhu_A0_quantile, axis=0)
                for ann_data in annual_A0_arrays
            ]
        )

        # Calculate the fractional allocation of GPP to LAI using an annual quantile of
        # the daily assimilation values.
        # TODO - apply only to input years
        self.lai_to_gpp_ratio_m: NDArray[np.floating] = (
            fapar_limitation.lai_max / daily_A0_quantiles
        )
        """Annual values for the fractional allocation of potential GPP to leaf area
        index (:math:`m`)."""

        # Duplicate m ratio for each day and calculate daily mu value as m * daily molar
        # assimilation:
        steady_state_lai = (
            self.lai_to_gpp_ratio_m[year_index] * daily_potential_assimilation
        )

        # Calculate the lag length for each site based on the AET/PET ratio
        aridity_lag_days = np.round(
            np.clip(
                aet_pet_ratio * phenology_const.zhu_lagcoef,
                0.0,
                phenology_const.zhu_lagmax,
            )
        ).astype(int)

        # NOTE: the Zhu code in plmodel_timeseries allows for steps in the input data
        #       values that are not at daily intervals and then corrects the lag length
        #       for this interval. This is not yet implemented.

        # Add the spin up at the start of the time series - the original implementation
        # expected annual blocks of data and just tiled the data to double the length
        # and hence give a single year spin up to the year of data. Here we add a fixed
        # block given by the spinup length at the start to allow multiple years.
        steady_state_with_spinup = np.concatenate(
            [steady_state_lai[:spinup_length], steady_state_lai],
            axis=0,
        )

        # The calculation of realised LAI uses mean values over the preceding lagging
        # period of N days:
        # * The numerator is the sum of the previous N observations, which can be
        #   calculated efficiently by taking the cumulative sum across all observations
        #   and subtracting the value N steps before. The initial period when that would
        #   run off the start of the array are handled by simply padding the cumulative
        #   sums with zero.
        # * The denominator is then the number of lagging days N, except for the first N
        #   observations which should be 1:N.
        # At the moment, because the sites are padded with site specific lags, the code
        # iterates over sites. There may be a fancy way to use numpy to pad all in one
        # go, but not obvious.

        # Get cumulative sums of LAI values along the time axis
        cumulative_lai = np.cumsum(steady_state_with_spinup, axis=0)

        # Create an output array to populate
        realised_lai = np.full_like(steady_state_with_spinup, fill_value=np.nan)

        # Record the number of observations along the time axis in the spinup array
        n_observations = steady_state_with_spinup.shape[0]

        # Get an index over the combinations of other axes - i.e. over sites
        site_index = np.ndindex(cumulative_lai[0].shape)

        for idx in site_index:
            full_idx = (slice(None), *idx)
            # Get the site specific lag
            lag_length = aridity_lag_days[*idx].item() + 1
            # Get the numerator the cumulative sum minus the same values but zero padded
            # on the left to the lag length and then truncated back to the same shape
            numerator = (
                cumulative_lai[full_idx]
                - np.pad(cumulative_lai[full_idx], (lag_length, 0))[:n_observations]
            )
            # Get the denominator
            denominator = np.minimum(np.arange(n_observations), lag_length)
            # Store the result
            realised_lai[full_idx] = numerator / denominator

        # Remove the spin up data along the first axis
        realised_lai = realised_lai[spinup_length:]

        self.steady_state_lai = steady_state_lai
        self.realised_lai = realised_lai


class Phenology:
    r"""Estimating daily time series of leaf area index (LAI).

    The maximum fAPAR and LAI for a year (see
    :class:`~pyrealm.phenology.fapar_limitation.FaparLimitation`) can be combined
    with estimates of daily potential assimilation to generate a time series of expected
    LAI that captures the annual phenology for a site. This class provides methods to
    calculate two time series for LAI:

    * :math:`L_s` - the steady state LAI that would be achieved if plants can instantly
      convert assimilation into leaf area.
    * :math:`L_r` - the realised LAI, incorporating lags to capture realistic delays in
      deploying leaf area.

    The class supports different methods for estimating these time series and also
    provides the :meth:`Phenology.from_pmodel` method that calculates the required
    daily assimilation values from a fitted P Model.

    Args:
        fapar_limitation: A
            :class:`~pyrealm.phenology.fapar_limitation.FaparLimitation` instance
            providing estimates of maximum annual fAPAR and LAI.
        daily_potential_assimilation: A daily time series of potential assimilation
            (:math:`A_0`, mol m-2 day-1).
        datetimes: The datetimes of the observations of :math:`A_0`, used to match
            observations to the appropriate annual values.
        method: A string selecting the method to be used to estimate LAI.
    """

    __experimental__ = True

    def __init__(
        self,
        fapar_limitation: FaparLimitation,
        daily_potential_assimilation: NDArray[np.floating],
        datetimes: NDArray[np.datetime64],
        method: str = "zhou",
        **kwargs: Any,
    ):
        """Constructor method for Phenology."""

        # Experimental class
        warn_experimental(self.__class__.__name__)

        # Check the array input shapes
        check_input_shapes(
            daily_potential_assimilation,
            datetimes,
            *kwargs.values(),
        )

        # Store datetimes and daily potential assimilation - this is primarily to
        # support the from_pmodel method, where the daily values are interpolated or
        # aggregated and need to be accessed.
        self.datetimes: NDArray[np.datetime64] = datetimes
        self.daily_potential_assimilation: NDArray[np.floating] = (
            daily_potential_assimilation
        )

        # Check the datetimes provide ordered daily resolution observations - don't
        # insist on daily precision but check that second level representations of the
        # days is consistent with daily observations and strictly increases.
        datetimes = datetimes.astype("datetime64[s]")
        datetime_spacing = np.diff(datetimes)

        if not np.all(np.equal(datetime_spacing, 86400)):
            raise ValueError(
                "The datetimes argument must provide timestamps of a "
                "complete increasing daily time series"
            )

        # Set up the fAPAR limitation method to be used.
        self.method: str = method
        if method not in PHENOLOGY_METHOD_CLASS_REGISTRY:
            raise ValueError(f"Unknown FaparLimitation method: {method}")

        # Get the years of observations and check they are all represented in the
        # FaparLimitation object
        datetimes_year = datetimes.astype("datetime64[Y]")
        observation_years = np.unique(datetimes_year)

        missing_years = set(observation_years).difference(fapar_limitation.years)
        if missing_years:
            raise ValueError(
                f"The observation datetimes include years that are not included in the "
                f"fapar_limitation data: {', '.join([str(y) for y in missing_years])}"
            )

        # Store fapar_limitation class
        self.fapar_limitation = fapar_limitation
        """The annual maximum fAPAR and LAI data used in the model."""

        # Calculate the index of each observation in the FaparLimitation years.
        # This uses a shortcut to avoid looking up using np.where or np.searchsorted:
        # * The FaparLimitation.years are a sequence of values: N, N+1, N+2, ..., N+M
        # * The datetime_years are now validated to lie in N, ..., N+M
        # * If we take the first year in FaparLimitation.years as an integer then that
        #   provides an offset from zero for the indexing of the year sequence, and we
        #   can subtract that from the observation years to get the index of each
        #   observation.
        year_index = datetimes_year.astype("int") - fapar_limitation.years[0].astype(
            "int"
        )

        # Get an instance of the method class from the registry
        phenology_method: PhenologyMethodABC = PHENOLOGY_METHOD_CLASS_REGISTRY[method](
            fapar_limitation=self.fapar_limitation,
            daily_potential_assimilation=daily_potential_assimilation,
            datetimes=datetimes,
            year_index=year_index,
            **kwargs,
        )

        self.steady_state_lai: NDArray[np.floating] = phenology_method.steady_state_lai
        self.realised_lai: NDArray[np.floating] = phenology_method.realised_lai

    @classmethod
    def from_pmodel(
        cls,
        pmodel: PModelABC,
        fapar_limitation: FaparLimitation,
        datetimes: NDArray[np.datetime64] | None = None,
        gpp_penalty_factor: NDArray[np.floating] | None = None,
        **kwargs: Any,
    ) -> Phenology:
        r"""Calculate daily phenology from a P Model and other inputs.

        TBD.

        Args:
            pmodel: A :class:`pyrealm.pmodel.pmodel.PModel` or
                :class:`pyrealm.pmodel.pmodel.SubdailyPModel` instance, fitted with
                ``fapar`` fixed at one.
            fapar_limitation: A FaparLimitation object providing the maximum annual LAI
                and fAPAR.
            datetimes: An array giving the datetimes of observations.
            gpp_penalty_factor: A GPP penalty factor.
            **kwargs: Additional arguments.
        """

        # Check the datetimes - should they be taken from the AcclimationModel of the
        # SubdailyPModel or are they required for standard PModels?
        if isinstance(pmodel, SubdailyPModel):
            if datetimes is not None:
                raise ValueError(
                    "Observation datetimes are not required with SubdailyPModel "
                    "inputs, the acclimation model datetimes are used."
                )
            datetimes = pmodel.acclim_model.datetimes
            daily_timestamps, daily_gpp = pmodel._get_daily_gpp()

        elif isinstance(pmodel, PModel):
            if datetimes is None:
                raise ValueError(
                    "Observation datetimes are required with PModel inputs."
                )
            daily_timestamps, daily_gpp = pmodel._get_daily_gpp(datetimes=datetimes)

        # Scale daily GPP in µmol m2 s up to daily molar assimilation.
        daily_potential_assimilation = (
            daily_gpp * 60 * 60 * 24 * 1e-6
        ) / pmodel.core_const.k_c_molmass

        return cls(
            fapar_limitation=fapar_limitation,
            datetimes=daily_timestamps,
            daily_potential_assimilation=daily_potential_assimilation,
            **kwargs,
        )
