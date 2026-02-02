"""Classes to compute phenological time series of Leaf Area Index.

The :class:`PhenologyNew` class and the :meth:`PhenologyNew.from_pmodel` are designed to
work with inputs that can have multiple dimensions. The first axis is _always_ assumed
to represent a time series of daily observations of potential assimilation. If all the
arrays are one dimensional, then this is a time series for a single site; if they are
three dimensional then these are observations for a grid of sites. Usually all array
inputs will have the same shape but note the following instances where you might need to
take care with array broadcasting.

* The climatological aridity index is very likely to be constant through time.
    If so, the array should have a singleton first dimension to broadcast an
    observation per site across observations. For example, with `(10, 3, 3)` data
    (10 years for a 3x3 grid of sites), the aridity index could be provided as
    `(1,3,3)` to broadcast the aridity index across each year. It could also use a
    single scalar value to use the same aridity index for all sites.
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
from pyrealm.phenology.fapar_limitation_new import FaparLimitationNew
from pyrealm.pmodel.pmodel import PModel, PModelABC, SubdailyPModel

PHENOLOGY_METHOD_CLASS_REGISTRY: dict[str, type[PhenologyMethodABC]] = {}
"""A registry for classes implementing different methods for estimating phenology.

Different implementations of the calculation of LAI phenology must all be subclasses of
the :class:`~pyrealm.phenology.phenology.New` abstract base class.
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
    """A tuple of attributes to be reported in the PhenologyNew.summarize() output
    for the method."""

    requires: tuple[str, ...]
    """A tuple of any additional variables required to create a PhenologyNew
    instance using the method."""

    @abstractmethod
    def __init__(
        self,
        fapar_limitation: FaparLimitationNew,
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
    """Blah blah blah."""

    __experimental__ = True

    attrs = (
        ("steady_state_lai", "-"),
        ("realised_lai", "-"),
    )

    requires = tuple()

    def __init__(
        self,
        fapar_limitation: FaparLimitationNew,
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

        # Calculate the LAI to GPP ratio
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
    """Blah blah blah."""

    __experimental__ = True

    attrs = (
        ("steady_state_lai", "-"),
        ("realised_lai", "-"),
    )

    requires = ("aet_pet_ratio",)

    def __init__(
        self,
        fapar_limitation: FaparLimitationNew,
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

        # Calculate the ratio of LAI to the annual quantile of daily assimilation.
        # TODO apply only to input years
        self.lai_to_gpp_ratio_m: NDArray[np.floating] = (
            fapar_limitation.lai_max / daily_A0_quantiles
        )
        """Annual values of the steady state ratio of leaf area index to potential GPP
        (:math:`m`)"""

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
            # Get the site specific lag
            lag_length = aridity_lag_days[*idx].item() + 1
            # Get the numerator the cumulative sum minus the same values but zero padded
            # on the left to the lag length and then truncated back to the same shape
            numerator = (
                cumulative_lai[idx]
                - np.pad(cumulative_lai[idx], (lag_length, 0))[:n_observations]
            )
            # Get the denominator
            denominator = np.minimum(np.arange(n_observations), lag_length)
            # Store the result
            realised_lai[idx] = numerator / denominator

        # Remove the spin up data along the first axis
        realised_lai = realised_lai[spinup_length:]

        self.steady_state_lai = steady_state_lai
        self.realised_lai = realised_lai


class PhenologyNew:
    """Phenology calculation."""

    __experimental__ = True

    def __init__(
        self,
        fapar_limitation: FaparLimitationNew,
        daily_potential_assimilation: NDArray[np.floating],
        datetimes: NDArray[np.datetime64],
        method: str = "zhou",
        **kwargs: Any,
    ):
        # Experimental class
        warn_experimental(self.__class__.__name__)

        # Check the array input shapes
        check_input_shapes(
            daily_potential_assimilation,
            datetimes,
            *kwargs.values(),
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
        fapar_limitation: FaparLimitationNew,
        datetimes: NDArray[np.datetime64] | None = None,
        **kwargs: Any,
    ) -> PhenologyNew:
        r"""Calculate daily phenology from a P Model and other inputs.

        TBD.

        Args:
            pmodel: A :class:`pyrealm.pmodel.pmodel.PModel` or
                :class:`pyrealm.pmodel.pmodel.SubdailyPModel` instance, fitted with
                ``fapar`` fixed at one.
            fapar_limitation: A FaparLimitation object providing the maximum annual LAI
                and fAPAR.
            datetimes: An array giving the datetimes of observations.
            **kwargs: Additional arguments.
        """

        daily_gpp: NDArray[np.floating]

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

        # The datetimes argument cannot now be None, so mute type error.
        return cls(
            fapar_limitation=fapar_limitation,
            datetimes=daily_timestamps,
            daily_potential_assimilation=daily_potential_assimilation,
            kwargs=kwargs,
        )
