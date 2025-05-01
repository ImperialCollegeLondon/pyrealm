"""Class to compute the fAPAR_max and annual peak Leaf Area Index (LAI)."""

from collections.abc import Callable
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from pyrealm.constants import PhenologyConst
from pyrealm.core.utilities import check_input_shapes
from pyrealm.pmodel import AcclimationModel, PModel, SubdailyPModel


def check_datetimes(datetimes: NDArray[np.datetime64]) -> None:
    """Check that the datetimes are in a valid format."""

    deltas = datetimes[1:] - datetimes[:-1]
    unique_deltas = np.unique(deltas)

    # check that we have uniformly sampled data
    if np.size(unique_deltas) > 1:
        raise ValueError("datetimes are not evenly spaced.")

    dates = datetimes.astype("datetime64[D]")
    unique_dates, date_counts = np.unique(dates, return_counts=True)
    if not date_counts.max() == date_counts.min():
        raise ValueError("Differing date counts per day")

    obs_per_date = date_counts.max()

    # Data needs to start in northern or southern hemisphere midwinter
    first_month = unique_dates[0].astype("datetime64[M]").astype(str)
    if not (first_month.endswith("01") | first_month.endswith("07")):
        raise ValueError("Data does not start in January or July.")

    ## This does not work for fortnightly data
    # no_leapdays = leapdays(
    #    int(str(datetimes[0].astype("datetime64[Y]"))),
    #    int(str(datetimes[-1].astype("datetime64[Y]"))),
    # )
    #
    # year_remainder = len(unique_dates) % 365
    # check that we have the right number of leap days
    #    if year_remainder > no_leapdays:
    #        raise ValueError("Datetimes do not cover full years.")

    if obs_per_date > 1:
        # subdaily

        # Check that the number of observations per day is evenly divisible by the
        # number of seconds in a day (should already have led to differing date counts).
        day_remainder = (24 * 60 * 60) % obs_per_date
        if day_remainder:
            raise ValueError("Datetime spacing is not evenly divisible into a day.")


class AnnualValueCalculator:
    """An annual value calculation class.

    This class is used to calculate annual mean and sum values from time series data. An
    instance is created using either an array of datetimes for time series data or an
    AcclimationModel instance that has been used to fit a SubdailyPModel, which provides
    an already validated set of datetimes at subdaily temporal resolutions. These
    datetimes are used to calculates indices along the time dimension that will split
    data arrays into annual blocks, allowing the instance to be used to calculate annual
    values for multiple variables.

    When created, the class also takes a boolean array indicating which observations are
    part of the growing season.

    Temporal resolution
    -------------------

    The datetimes provided to the the class are taken to indicate the start time of each
    observation period, and the total durations of these observations are used to check
    that complete data is provided for a set of years. Where the temporal resolution is
    a number of days (e.g. weekly or dekadal data), which may not map precisely onto
    year boundaries, a tolerance equal to the length of one interval is allowed in
    matching observation durations within a year to the actual year duration. However,
    where the datetimes suggest a monthly, daily or subdaily resolution, the total
    duration of observations within a year expected to map precisely onto the length of
    each year.


    .. TODO::

        * Should this be more precise about year ends - calculate the actual weighted
          values across years so that weekly data also has to be complete.

    """

    def __init__(
        self,
        timing: AcclimationModel | NDArray[np.datetime64],
        growing_season: NDArray[np.bool_],
    ):
        # Attribute definitions
        self.datetimes: NDArray[np.datetime64]
        """TBD"""
        self.year_breaks: NDArray[np.int_]
        """TBD"""
        self.duration_seconds: NDArray[np.timedelta64]
        """TBD"""
        self.growing_season: NDArray[np.bool_]
        """TBD"""
        self.growing_season_by_year: list[NDArray[np.bool_]]
        """TBD"""

        # Sanity checks on datetimes
        if not (
            isinstance(timing, AcclimationModel)
            or (
                isinstance(timing, np.ndarray)
                and np.issubdtype(timing.dtype, np.datetime64)
            )
        ):
            raise ValueError(
                "The timings argument must be an AcclimationModel "
                "or an array of datetime64 values"
            )

        # Check the coverage of years and set the required precision
        if isinstance(timing, AcclimationModel):
            # AcclimationModel provides subdaily data, so need full days with no
            # tolerance of partial annual data.
            self.datetimes = timing.datetimes
            datetimes_seconds = self.datetimes.astype("datetime64[s]")
            self.duration_seconds = np.append(
                np.diff(datetimes_seconds), timing.spacing
            )
            year_tolerance_seconds = 0
        else:
            # Datetime inputs could be any frequency from subdaily to monthly.
            # Convert time to seconds precision
            self.datetimes = timing
            datetimes_seconds = timing.astype("datetime64[s]")

            # Get the intervals and figure out the duration of the last observation
            duration_seconds = np.diff(datetimes_seconds)
            intervals: NDArray = np.unique(duration_seconds)

            # Check for constant gaps _or_ months
            if len(intervals) == 1:
                # Constant intervals - TODO check here for daily?
                duration_last_observation = duration_seconds[0]
                year_tolerance_seconds = duration_last_observation
            elif (len(intervals) > 1) and {
                2419200,
                2505600,
                2592000,
                2678400,
            }.issuperset(intervals.astype("int")):
                # Dealing with monthly data so calculate the length of the last month
                next_month = (
                    self.datetimes[-1].astype("datetime64[M]") + np.timedelta64(1, "M")
                ).astype("datetime64[s]")
                duration_last_observation = next_month - datetimes_seconds[-1]
                year_tolerance_seconds = 0
            else:
                # TODO - how to handle bimonthly or quarterly observations.
                raise ValueError(
                    "Datetimes must be monthly or use a constant interval."
                )

            self.duration_seconds = np.append(
                duration_seconds, duration_last_observation
            )

        # Calculate the expected number of seconds in each year appearing in the time
        # series to calculate the year coverage
        datetimes_by_year = self.datetimes.astype("datetime64[Y]")
        years = np.unique(datetimes_by_year)
        expected_year_duration_seconds = np.diff(
            np.append(years, years[-1] + np.timedelta64(1, "Y")).astype("datetime64[s]")
        )

        # Get the points along the time axis at which year changes
        self.year_breaks = np.where(np.diff(datetimes_by_year))[0] + 1

        actual_year_duration_seconds = np.array(
            [v.sum() for v in np.split(self.duration_seconds, self.year_breaks)]
        )
        if not np.allclose(
            actual_year_duration_seconds,
            expected_year_duration_seconds,
            atol=year_tolerance_seconds,
        ):
            raise ValueError(
                "Data timings do not cover complete years to within tolerance"
            )

        # Sanity checks on growing season
        if not np.issubdtype(growing_season.dtype, np.bool_):
            raise ValueError("Growing season data is not an array of boolean values")

        if not self.datetimes.shape == growing_season.shape:
            raise ValueError(
                "Growing season data is not the same shape as the timing data"
            )

        # Split the growing season up into a list of subarrays by year
        self.growing_season = growing_season
        self.growing_season_by_year = np.split(growing_season, self.year_breaks)

    def get_annual_values(
        self,
        values: NDArray[np.float64],
        function: str = "mean",
        within_growing_season: bool = True,
    ) -> NDArray[np.float64]:
        """Get annual summary stats from an array of values.

        Args:
            values: The data to summarize by year
            function: The required summary statistic
            within_growing_season: Should the statistic only include values within the
                growing season.
        """

        if values.shape != self.datetimes.shape:
            raise ValueError("Values array shape does not match datetimes")

        # Split the daily values into subarrays using the year breaks
        values_by_year = np.split(values, self.year_breaks)

        # TODO - could possibly pass a ufunc in directly as the function argument - more
        # general and cleaner - but I can't work out the typing right now! Could simply
        # use Callable as below, but really we want to specifically restrict to unary
        # ufuncs.
        ufunc: Callable
        match function:
            case "mean":
                ufunc = np.mean
            case "sum":
                ufunc = np.sum
            case _:
                raise ValueError("The function argument '{function}' is not known")

        if not within_growing_season:
            return np.array([ufunc(vals) for vals in values_by_year])

        # Iterate over the paired subarrays of values and growing season, applying the
        # function to each subset of values within the growing season.
        return np.array(
            [
                ufunc(vals[gs])
                for vals, gs in zip(values_by_year, self.growing_season_by_year)
            ]
        )


class AnnualValueCalculatorMarkII:
    """An annual value calculation class.

    This class is used to calculate annual means and totals from time series data. An
    instance is created by providing a set of timings for the times series data, either
    as an an array of datetimes or as an AcclimationModel instance from a
    SubdailyPModel, which provides validated datetimes at subdaily temporal resolutions.

    The calculation process accounts for observations that span year boundaries, such as
    fortnightly data, by calculating the duration of each observation within each year.
    The process also handles unequal sampling intervals - such as monthly data - by
    calculating the actual duration of observations. However, the duration of
    the last interval is unknown and so an explicit endpoint must be provided.

    The indexing of annual subsets of observations and observation weights are
    calculated when the class is created and then used by the get_annual_means and
    get_annual_totals methods.
    """

    def __init__(
        self,
        timing: AcclimationModel | NDArray[np.datetime64],
        growing_season: NDArray[np.bool_] | None = None,
        endpoint: np.datetime64 | None = None,
    ):
        # Attribute definitions
        self.datetimes: NDArray[np.datetime64]
        """The datetimes of observations taking from the initial timings"""
        self.endpoint: np.datetime64
        """A datetime giving of the end of the last observation."""
        self.growing_season: NDArray[np.bool_]
        """The initial input array of growing season data."""

        self.indexing: list[tuple[int, int]] = []
        """Pairs of integers giving start and end indices to extract consecutive years
        of data from the time series."""
        self.duration_weights: list[NDArray[np.float64]] = []
        """A list of arrays giving the number of seconds that each observation
        within a year contributes to that year."""
        self.fractional_weights: list[NDArray[np.float64]] = []
        """A list of arrays giving the fraction of each observation within a year that
        falls in the year."""
        self.growing_season_by_year: list[NDArray[np.bool_]]
        """A list of arrays giving the growing season subarrays for each year."""
        self.year_completeness: NDArray[np.float64]
        """Provides the fractional coverage of observations for each year."""

        # Sanity checks on datetimes
        if not (
            isinstance(timing, AcclimationModel)
            or (
                isinstance(timing, np.ndarray)
                and np.issubdtype(timing.dtype, np.datetime64)
            )
        ):
            raise ValueError(
                "The timings argument must be an AcclimationModel "
                "or an array of datetime64 values"
            )

        if isinstance(timing, AcclimationModel):
            # AcclimationModel by construction provides subdaily data with equal spacing
            self.datetimes = timing.datetimes.astype("datetime64[s]")
            duration_last_observation = timing.spacing.astype("datetime64[s]")
        else:
            # Pure datetime inputs could be any frequency from subdaily to monthly, and
            # some frequencies could be of differing lengths (monthly being a good
            # example)

            # Convert time to seconds precision
            self.datetimes = timing.astype("datetime64[s]")

            # Get the intervals in seconds and see if they are strictly increasing and
            # then if the spaing is consistent. If the spacing is not consistent, then
            # require an endpoint for the observations.
            duration_seconds = np.diff(self.datetimes)

            if not np.all(duration_seconds > 0):
                raise ValueError("The timing values are not strictly increasing")

            intervals: NDArray = np.unique(duration_seconds)

            if len(intervals) == 1:
                # Constant intervals
                duration_last_observation = duration_seconds[0]
            else:
                if endpoint is None:
                    raise ValueError(
                        "The timings values are not equally spaced: provide an "
                        "explicit endpoint"
                    )

                if endpoint <= timing[-1]:
                    raise ValueError(
                        "The end_datetime value must be greater than the "
                        "last timing value"
                    )

                duration_last_observation = (endpoint - self.datetimes[-1]).astype(
                    "timedelta64[s]"
                )

        # Sanity checks on growing season
        if growing_season is None:
            growing_season = np.ones_like(self.datetimes, dtype=np.bool_)
        else:
            if not np.issubdtype(growing_season.dtype, np.bool_):
                raise ValueError(
                    "Growing season data is not an array of boolean values"
                )

            if not self.datetimes.shape == growing_season.shape:
                raise ValueError(
                    "Growing season data is not the same shape as the timing data"
                )
        # Store the growing season data
        self.growing_season = growing_season

        # Record the endpoint to get the total timespan of the data and hence the
        # duration of each observation
        self.endpoint = self.datetimes[-1] + duration_last_observation
        timespan = np.append(self.datetimes, self.endpoint)
        observation_durations = np.diff(timespan)

        # Now get the datetimes of the start of each of year included in the data
        years = np.unique(self.datetimes.astype("datetime64[Y]"))

        # Add trailing year to handle partial data at the end.
        if years[-1] < self.datetimes[-1]:
            years = np.append(years, years[-1] + np.timedelta64(1, "Y"))

        # Convert to second precision and find where they occur in the timespan
        years = years.astype("datetime64[s]")
        year_change_indices = np.searchsorted(timespan, years)

        # Now assign the duration of each observation across years, allowing for year
        # changes that occur during an observation, storing the indices of subsets and
        # the weighting to be used with values.

        # Iterate over pairs of year dates and indices
        for (lower, upper), (lower_index, upper_index) in zip(
            pairwise(years), pairwise(year_change_indices)
        ):
            # Get the initial set of datetimes within the year
            year_datetimes = timespan[lower_index:upper_index]

            # If the upper index is not to the end of the time series, then append the
            # the closing time for the current year at the end and extend the sample to
            # include the next value.
            #
            # Note here that the indexing of the final observation does not require
            # special handling because np.searchsorted returns a last index _beyond_ the
            # end of the timespan, so will automatically include the last observation.
            if upper_index < len(timespan):
                year_datetimes = np.append(year_datetimes, upper)

            # If the first observation is not the precise start of the year _and_ we are
            # not on the first year of data, then we also need to shift lower_index down
            # to include partial data from the previous observation and add the year
            # start to the internal datetimes
            if (year_datetimes[0] != lower) and (lower_index > 0):
                lower_index -= 1
                year_datetimes = np.insert(year_datetimes, 0, lower)

            # Calculate the duration of the observations within the year span
            internal_year_durations = np.diff(year_datetimes)

            # Divide the internal duration through by the actual observation durations
            # to get fractional weights.
            fractional_duration = (
                internal_year_durations / observation_durations[lower_index:upper_index]
            )

            # Store the indices and weights
            self.indexing.append((int(lower_index), int(upper_index)))
            self.duration_weights.append(internal_year_durations)
            self.fractional_weights.append(fractional_duration)

        # Split the growing season up into a list of subarrays by year
        self.growing_season_by_year = [
            growing_season[lower:upper] for lower, upper in self.indexing
        ]

        # Populate the year completeness
        self.year_completeness = np.array(
            [np.sum(v) for v in self.duration_weights]
        ) / np.diff(years)

    def _split_values_by_year(
        self, values: NDArray[np.float64]
    ) -> list[NDArray[np.float64]]:
        """Validates and splits value arrays.

        Args:
            values: An array of values.
        """

        if values.shape != self.datetimes.shape:
            raise ValueError("Values array shape does not match datetimes")

        # Split the daily values into subarrays for each year
        return [values[lower:upper] for lower, upper in self.indexing]

    def get_annual_means(
        self,
        values: NDArray[np.float64],
        within_growing_season: bool = True,
    ) -> NDArray[np.floating]:
        """Get annual means from an array of values.

        Average values are calculated weighted by the __duration__ of each observation,
        including weighting partial observations across year boundaries. If
        ``within_growing_season`` is ``True``, the weights for observations not
        identified as growing season values are set to zero.

        Args:
            values: The data to summarize by year
            within_growing_season: Should the mean only include values within the
                growing season.
        """

        values_by_year = self._split_values_by_year(values)

        # Averages use _duration_ weights
        if within_growing_season:
            weights = [
                wght * gs
                for wght, gs in zip(self.duration_weights, self.growing_season_by_year)
            ]
        else:
            weights = self.duration_weights

        return np.array(
            [
                np.average(vals, weights=wghts)
                for vals, wghts in zip(values_by_year, weights)
            ]
        )

    def get_annual_totals(
        self,
        values: NDArray[np.float64],
        within_growing_season: bool = True,
    ) -> NDArray[np.floating]:
        """Get annual totals from an array of values.

        The contribution of each observation to the total is weighted by the
        __fractional__ duration of each observation within each year in order to
        partition the sum across years correctly. If ``within_growing_season`` is
        ``True``, the weights for observations not identified as growing season values
        are set to zero.

        Args:
            values: The data to summarize by year
            within_growing_season: Should the mean only include values within the
                growing season.
        """

        values_by_year = self._split_values_by_year(values)

        # Totals use _fractional_ weights
        if within_growing_season:
            weights = [
                wght * gs
                for wght, gs in zip(
                    self.fractional_weights, self.growing_season_by_year
                )
            ]
        else:
            weights = self.fractional_weights

        return np.array(
            [np.sum(vals * wghts) for vals, wghts in zip(values_by_year, weights)]
        )


def get_annual(
    x: NDArray,
    datetimes: NDArray[np.datetime64],
    growing_season: NDArray[np.bool],
    method: str,
) -> NDArray:
    """Computes an array of the annual total or mean of an entity x given datetimes.

    Args:
        x: Array of values to be converted to annual values. Should be either daily (
            same datetimes as growing_season) or subdaily (same datetimes as datetimes
            array)
        datetimes: Datetimes of the measurements as np.datetime64 arrays.
        growing_season: Bool array of days, indicating whether they are ain growing
            season or not.
        method: Either "total" (sum all values of the year) or "mean" (take the mean
            of all values of the year)
    """

    # Extract years from datetimes
    all_years = datetimes.astype("datetime64[Y]")

    if len(x) == len(growing_season):  # this is daily data
        daily_x = x
        n_days = len(x)
        years_by_day = all_years.view()
        obs_per_day = int(len(datetimes) / n_days)
        years_by_day.shape = tuple([n_days, obs_per_day, *list(all_years.shape[1:])])
    elif len(x) == len(datetimes):  # this is subdaily data
        # Create scaler object to handle conversion between scales
        scaler = AcclimationModel(datetimes)
        scaler.set_nearest(np.timedelta64(12, "h"))
        # Convert values to daily to match with growing_season
        daily_x = scaler.get_daily_means(x)
        years_by_day = scaler.get_window_values(np.asarray(all_years))
    else:
        raise ValueError("Input array does not fit datetimes nor growing_season array")

    # Which years are present?
    years = np.unique(all_years)

    # Compute annual totals or means, only taking into account the days which are in
    # growing season.

    annual_x = np.zeros(len(years))

    if method == "total":
        for i in range(len(years)):
            annual_x[i] = sum(
                daily_x[growing_season & (years_by_day[:, 0] == years[i])]
            )
    elif method == "mean":
        for i in range(len(years)):
            annual_x[i] = np.mean(
                daily_x[growing_season & (years_by_day[:, 0] == years[i])]
            )
    else:
        raise ValueError("No valid method given for annual values")

    return annual_x


def daily_to_subdaily(
    x: NDArray,
    datetimes: NDArray[np.datetime64],
) -> NDArray:
    """Broadcasts an array of the entity x from daily values to subdaily values.

    Args:
        x: Array of daily values.
        datetimes: Subdaily datetimes as np.datetime64 array.
    """

    n_days = len(x)
    obs_per_day = int(len(datetimes) / n_days)

    subdaily_x = np.repeat(x, obs_per_day)

    return subdaily_x


class FaparLimitation:
    r"""FaparLimitation class to compute fAPAR_max and LAI.

    This class takes the annual total potential GPP and precipitation, the annual mean
    CA, Chi and VPD during the growing season, as well as the aridity index and some
    constants to compute the annual peak fractional absorbed photosynthetically active
    radiation ( fAPAR_max) and annual peak Leaf Area Index (LAI).

    .. todo::

        * Allow for other timescales than daily or subdaily (e.g. monthly)
        * Daily conditions are taken as noon values - might need to relax this.
        * Growing season definition -  users currently need to provide their own growing
          season definition in `from_pmodel`
    """

    def _check_shapes(self) -> None:
        """Internal class to check all the input arrays have the same size."""

        check_input_shapes(
            self.annual_total_potential_gpp,
            self.annual_mean_ca,
            self.annual_mean_chi,
            self.annual_mean_vpd,
            self.annual_total_precip,
            self.aridity_index,
        )

        # assert len(self.annual_total_potential_gpp) == len(self.annual_mean_ca)
        # assert len(self.annual_mean_ca) == len(self.annual_mean_chi)
        # assert len(self.annual_mean_chi) == len(self.annual_mean_vpd)
        # assert len(self.annual_mean_vpd) == len(self.annual_total_precip)
        # if np.shape(self.aridity_index) != ():
        #     assert len(self.annual_total_precip) == len(self.aridity_index)

    def __init__(
        self,
        annual_total_potential_gpp: NDArray[np.float64],
        annual_mean_ca: NDArray[np.float64],
        annual_mean_chi: NDArray[np.float64],
        annual_mean_vpd: NDArray[np.float64],
        annual_total_precip: NDArray[np.float64],
        aridity_index: NDArray[np.float64],
    ) -> None:
        r"""Annual peak fractional absorbed photosynthetically active radiation (fAPAR).

        Computes fAPAR_max as minimum of a water-limited and an energy-limited
        quantity.

        Args:
            annual_total_potential_gpp: Aka A_0, the annual sum of potential GPP.
              Potential GPP would be achieved if fAPAR = 1. [mol m^{-2} year^{-1}]
            annual_mean_ca: Ambient CO2 partial pressure. [Pa]
            annual_mean_chi: Annual mean ratio of leaf-internal CO2 partial pressure to
               c_a during the growing season (>0℃). [Pa]
            annual_mean_vpd: Aka D, annual mean vapour pressure deficit (VPD) during the
              growing season (>0℃). [Pa]
            annual_total_precip: Aka P, annual total precipitation. [mol m^{-2}
             year^{-1}]
            aridity_index: Aka AI, climatological estimate of local aridity index.
        """

        self.annual_total_potential_gpp = annual_total_potential_gpp
        """Aka A_0, the annual sum of potential GPP. [mol m^{-2} year^{-1}]"""
        self.annual_mean_ca = annual_mean_ca
        """Ambient CO2 partial pressure. [Pa]"""
        self.annual_mean_chi = annual_mean_chi
        """Annual mean ratio of leaf-internal CO2 partial pressure to c_a during the 
        growing season (>0℃). [Pa]"""
        self.annual_mean_vpd = annual_mean_vpd
        """Annual mean vapour pressure deficit (VPD) during the growing season [Pa]"""
        self.annual_total_precip = annual_total_precip
        """Annual total precipitation. [mol m^{-2} year^{-1}]"""
        self.aridity_index = aridity_index
        """Climatological estimate of local aridity index."""

        self._check_shapes()

        # Constants used for phenology computations
        self.phenology_const = PhenologyConst()

        #  f_0 is the ratio of annual total transpiration of annual total
        #  precipitation, which is an empirical function of the climatic Aridity Index
        #  (AI).
        a, b, c = self.phenology_const.f0_coefficients
        f_0 = a * np.exp(-b * np.log(aridity_index / c) ** 2)

        fapar_energylim = 1.0 - self.phenology_const.z / (
            self.phenology_const.k * annual_total_potential_gpp
        )
        fapar_waterlim = (
            f_0
            * annual_total_precip
            * annual_mean_ca
            * (1 - annual_mean_chi)
            / (1.6 * annual_mean_vpd * annual_total_potential_gpp)
        )

        self.fapar_max = np.minimum(fapar_waterlim, fapar_energylim)
        """Maximum fapar given water or energy limitation."""
        self.energy_limited = fapar_energylim < fapar_waterlim
        """Is fapar_max limited by water or energy."""
        self.annual_precip_molar = annual_total_precip
        """The annual precipitation in moles."""

        self.lai_max = -(1 / self.phenology_const.k) * np.log(1.0 - self.fapar_max)

    @classmethod
    def from_pmodel(
        cls,
        pmodel: PModel,
        growing_season: NDArray[np.bool],
        datetimes: NDArray[np.datetime64],
        precip: NDArray[np.float64],
        aridity_index: NDArray[np.float64],
        gpp_penalty_factor: NDArray[np.float64] | None = None,
    ) -> Self:
        r"""Get FaparLimitation from PModel input.

        Computes the input for fAPAR_max from the P Model and additional inputs.

        Args:
            pmodel: pyrealm.pmodel.PModel
            growing_season: Bool array indicating which times are within growing
              season by some definition and implementation.
            datetimes: Array of datetimes to consider.
            precip: Precipitation for given datetimes.
            aridity_index: Climatological estimate of local aridity index.
            gpp_penalty_factor: Penalty factor to be applied to pmodel.gpp
        """

        check_datetimes(datetimes)

        annual_total_potential_gpp = get_annual(
            pmodel.gpp, datetimes, growing_season, "total"
        )
        if gpp_penalty_factor is not None:
            annual_total_potential_gpp *= gpp_penalty_factor

        annual_mean_ca = get_annual(pmodel.env.ca, datetimes, growing_season, "mean")
        annual_mean_chi = get_annual(
            pmodel.optchi.chi, datetimes, growing_season, "mean"
        )
        annual_mean_vpd = get_annual(pmodel.env.vpd, datetimes, growing_season, "mean")
        annual_total_precip = get_annual(precip, datetimes, growing_season, "total")

        return cls(
            annual_total_potential_gpp,
            annual_mean_ca,
            annual_mean_chi,
            annual_mean_vpd,
            annual_total_precip,
            aridity_index,
        )

    @classmethod
    def from_subdailypmodel(
        cls,
        subdaily_pmodel: SubdailyPModel,
        growing_season: NDArray[np.bool],
        datetimes: NDArray[np.datetime64],
        precip: NDArray[np.float64],
        aridity_index: NDArray[np.float64],
        gpp_penalty_factor: NDArray[np.float64] | None = None,
    ) -> Self:
        r"""Get FaparLimitation from SubdailyPModel input.

        Computes the input for fAPAR_max from a Subdaily P Model and additional inputs.

        Args:
            subdaily_pmodel: pyrealm.pmodel.SubdailyPModel
            growing_season: Bool array indicating which times are within growing
              season by some definition and implementation.
            datetimes: Array of datetimes to consider.
            precip: Precipitation for given datetimes.
            aridity_index: Climatological estimate of local aridity index.
            gpp_penalty_factor: Penalty factor to be applied to subdaily_pmodel.gpp
        """

        check_datetimes(datetimes)

        annual_total_potential_gpp = get_annual(
            subdaily_pmodel.gpp,
            datetimes,
            growing_season,
            "total",
        )
        if gpp_penalty_factor is not None:
            annual_total_potential_gpp *= gpp_penalty_factor

        annual_mean_ca = get_annual(
            subdaily_pmodel.env.ca, datetimes, growing_season, "mean"
        )
        annual_mean_chi = get_annual(
            subdaily_pmodel.optchi.chi.round(5), datetimes, growing_season, "mean"
        )
        annual_mean_vpd = get_annual(
            subdaily_pmodel.env.vpd, datetimes, growing_season, "mean"
        )
        annual_total_precip = get_annual(precip, datetimes, growing_season, "total")

        return cls(
            annual_total_potential_gpp,
            annual_mean_ca,
            annual_mean_chi,
            annual_mean_vpd,
            annual_total_precip,
            aridity_index,
        )
