"""Class to compute the fAPAR_max and annual peak Leaf Area Index (LAI)."""

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from pyrealm.constants import PhenologyConst
from pyrealm.core.experimental import warn_experimental
from pyrealm.core.time_series import AnnualValueCalculator
from pyrealm.core.utilities import check_input_shapes
from pyrealm.pmodel import AcclimationModel, PModel


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

        # Check that the number of seconds in a day is evenly divisible by the
        # number of observations per day (should already have led to differing date
        # counts).
        day_remainder = (24 * 60 * 60) % obs_per_date
        if day_remainder:
            raise ValueError("Datetime spacing is not evenly divisible into a day.")


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
            annual_x[i] = np.sum(
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
    r"""Compute maximum annual fAPAR and LAI.

    This class calculates maximum annual fAPAR and LAI, following :cite:`cai:2025a`.
    The maximum annual fAPAR is calculated as the minimum of two terms capturing
    energy-limited and water limited fAPAR:

    .. math::

        fAPAR_{max} = \min{
                \left(1 - z / \left(k A_0 \right) \right),
                \left( c_a \left( 1 - \chi \right) / 1.6 D \right)
                \left( f_0 P / A_0 \right)
            }

    The maximum annual LAI is then calculated using Beer's law:

    .. math::

        LAI_{max} = - ( 1 / k ) \ln {1 - fAPAR_{max}}

    The :class:`~pyrealm.constants.phenology_const.PhenologyConst` class provides values
    for the following constants:

    * :math:`z` accounts for the growth and maintenance costs of leaves.
    * :math:`k` is the light extinction coefficient.
    * :math:`f_0` is is the ratio of annual total transpiration of annual total
      precipitation, calculated from the climatological aridity index (AI) (see
      :class:`PhenologyConst.calculate_f0<pyrealm.constants.phenology_const.PhenologyConst.calculate_f0>`).

    The other variables are the required arguments to the class defined below. The most
    common source of these variables is from a P Model, and the
    :meth:`~pyrealm.phenology.fapar_limitation.FaparLimitation.from_pmodel` method can
    be used to create an instance directly from a fitted P Model.

    Args:
        annual_total_potential_gpp: The annual sum of potential GPP (:math:`A_0,
            \text{mol C m}^{-2} \text{year}^{-1}`)
        annual_mean_ca: The ambient CO2 partial pressure during the growing season
            (:math:`c_a`, Pa)
        annual_mean_chi: The annual mean ratio of ambient to leaf CO2 partial during the
            growing season (:math:`\chi`, Pa)
        annual_mean_vpd: The annual mean vapour pressure deficit during the growing
            season (:math:`D`, Pa)
        annual_total_precip: The annual total precipitation (:math:`P, \text{mol m}^{-2}
            \text{year}^{-1}`)
        aridity_index: A climatological estimate of the local aridity index, calculated
            as the long term (typically 20 years) total PET over total precipitation
            (:math:`AI`, unitlesss)
        phenology_const: An instance of
            :class:`~pyrealm.constants.phenology_const.PhenologyConst`
    """

    __experimental__ = True

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

    def __init__(
        self,
        annual_total_potential_gpp: NDArray[np.float64],
        annual_mean_ca: NDArray[np.float64],
        annual_mean_chi: NDArray[np.float64],
        annual_mean_vpd: NDArray[np.float64],
        annual_total_precip: NDArray[np.float64],
        aridity_index: NDArray[np.float64],
        phenology_const: PhenologyConst = PhenologyConst(),
    ) -> None:
        # Experimental class
        warn_experimental("FaparLimitation")

        self.annual_total_potential_gpp = annual_total_potential_gpp
        r"""The annual sum of potential GPP 
        (:math:`A_0, \text{mol C m}^{-2} \text{year}^{-1}`)"""
        self.annual_mean_ca = annual_mean_ca
        r"""Ambient CO2 partial pressure during the growing season (:math:`c_a`, Pa)"""
        self.annual_mean_chi = annual_mean_chi
        r"""Annual mean ratio of ambient to leaf CO2 partial during the 
        growing season (:math:`\chi`, Pa)"""
        self.annual_mean_vpd = annual_mean_vpd
        r"""Annual mean vapour pressure deficit during the growing season (:math:`D`,
        Pa)"""
        self.annual_total_precip = annual_total_precip
        r"""Annual total precipitation
        (:math:`P, \text{mol m}^{-2} \text{year}^{-1}`)"""
        self.aridity_index = aridity_index
        r"""Climatological estimate of local aridity index (AI, unitless)"""

        self._check_shapes()

        # Make sure the aridity index is not zero
        if aridity_index <= 0:
            raise ValueError("The aridity index has to be positive.")

        # Constants used for phenology computations
        self.phenology_const = phenology_const

        #  f_0 is the ratio of annual total transpiration of annual total
        #  precipitation, which is an empirical function of the climatic Aridity Index
        #  (AI).
        f_0 = self.phenology_const.calculate_f0(aridity_index=self.aridity_index)

        # Calculate the energy and water limited terms.
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

        self.fapar_max: NDArray[np.floating] = np.minimum(
            fapar_waterlim, fapar_energylim
        )
        """Estimated annual maximum fAPAR (unitless)."""
        self.energy_limited: NDArray[np.bool_] = fapar_energylim < fapar_waterlim
        """Boolean array showing if annual :math:`fAPAR_{max}` is water or energy
        limited."""
        self.annual_precip_molar: NDArray[np.floating] = annual_total_precip
        """The annual total precipitation for each year (moles year-1)."""

        self.lai_max: NDArray[np.floating] = -(1 / self.phenology_const.k) * np.log(
            1.0 - self.fapar_max
        )
        """Estimated annual maximum LAI (unitless)"""

    @classmethod
    def from_pmodel(
        cls,
        pmodel: PModel,
        datetimes: NDArray[np.datetime64],
        growing_season: NDArray[np.bool],
        precip: NDArray[np.float64],
        aridity_index: NDArray[np.float64],
        gpp_penalty_factor: NDArray[np.float64] | None = None,
        phenology_const: PhenologyConst = PhenologyConst(),
    ) -> Self:
        r"""Create a FaparLimitation instance from a P Model and other inputs.

        The annual summary values of :math:`A_0, c_a, \chi` and :math:`D` used by the
        :meth:`~pyrealm.phenology.fapar_limitation.FaparLimitation` class can be taken
        directly from the predictions of a P Model. This method automatically extracts
        the required data from a fitted P Model and returns a ``FaparLimitation``
        instance.

        .. NOTE::

          The calculation of fAPAR limitation requires estimates of **potential** GPP,
          so the :class:`~pyrealm.pmodel.pmodel_environment.PModelEnvironment` instance
          used to fit the model **must** set ``fapar`` to be one.

        Some additional information is needed:

        * The calculation requires annual summaries of variables, so the ``datetimes``
          argument must be used to provide an array of datetimes for each observation.

        * The annual mean values :math:`c_a, \chi` and :math:`D` should be estimated
          during the growing season, so the ``growing_season`` argument must be used to
          provide a boolean value indicating which observations should be treated as in
          the growing season.

        * The calculation requires estimates of precipitation, so the ``precipitation``
          argument must provide estimates of total precipitation during each
          observations in moles of water per metre squared.

        * The calculation of the :math:`f_0` parameter requires estimates of site
          specific aridity index.

        The method accepts both standard and subdaily P Models and automatically uses
        the actual time intervals between observations to calculate the required
        weighted annual means and sums. This might lead to unexpected values: the yearly
        mean of monthly values :math:`1, 2, \dots, 12` would not be 6.5 because the
        monthly values are weighted according to the length of the month.

        Lastly, potential GPP is taken directly from the P Model instance. If you want
        to apply a post-hoc penalty factor to GPP (e.g. a water limitation factor), then
        you can optionally provide per-observation penalty estimates and they will be
        applied when calculating annual total potential assimilation.

        Args:
            pmodel: A :class:`pyrealm.pmodel.pmodel.PModel` or
                :class:`pyrealm.pmodel.pmodel.SubdailyPModel` instance, fitted with
                ``fapar`` fixed at one.
            datetimes: An array giving the datetimes of observations.
            growing_season: A boolean array indicating which observations are to be
                considered as part of the growing season.
            precip: An array of precipitation for each observation.
            aridity_index: A climatological estimate of local aridity index.
            gpp_penalty_factor: A post-hoc penalty factor to be applied to estimated
                GPP.
            phenology_const: An instance of
                :class:`~pyrealm.constants.phenology_const.PhenologyConst`
        """

        check_datetimes(datetimes)

        avc = AnnualValueCalculator(
            timing=datetimes,
            growing_season=growing_season,
        )

        # Get the total GPP for each observation
        # - also need to handle missing values, easier to take _mean_ annual value
        #   and scale it up to an annual total
        # - TODO - handle incompleteness - when do we stop estimating annual values from
        #   partial years (or at least warn about it)

        # Extract GPP and apply any observation level penalty factor
        total_gpp = pmodel.gpp
        if gpp_penalty_factor is not None:
            total_gpp *= gpp_penalty_factor

        # Calculate annual mean potential GPP and scale up to the year
        annual_mean_potential_gpp = avc.get_annual_means(total_gpp)
        annual_total_potential_gpp = (
            annual_mean_potential_gpp * (avc.year_n_days) * 86400 * 1e-6
        ) / pmodel.core_const.k_c_molmass

        # Calculate annual mean ca, chi and VPD within growing season
        annual_mean_ca = avc.get_annual_means(pmodel.env.ca, within_growing_season=True)
        annual_mean_chi = avc.get_annual_means(
            pmodel.optchi.chi, within_growing_season=True
        )
        annual_mean_vpd = avc.get_annual_means(
            pmodel.env.vpd, within_growing_season=True
        )

        # Calculate total annual precipitation
        annual_total_precip = avc.get_annual_totals(precip)

        return cls(
            annual_total_potential_gpp=annual_total_potential_gpp,
            annual_mean_ca=annual_mean_ca,
            annual_mean_chi=annual_mean_chi,
            annual_mean_vpd=annual_mean_vpd,
            annual_total_precip=annual_total_precip,
            aridity_index=aridity_index,
            phenology_const=phenology_const,
        )
