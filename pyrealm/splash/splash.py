"""The ``splash`` submodule provides the main SplashModel class for calculating
predictions under the SPLASH model.
"""  # noqa: D205

import warnings

import numpy as np
import pandas as pd  # type: ignore
from numpy.typing import NDArray

from pyrealm.constants import CoreConst
from pyrealm.core.bounds import BoundsChecker
from pyrealm.core.calendar import Calendar
from pyrealm.core.pressure import calc_patm
from pyrealm.core.time_series import broadcast_time
from pyrealm.core.utilities import check_input_shapes
from pyrealm.splash.evap import DailyEvapFluxes
from pyrealm.splash.solar import DailySolarFluxes


class SplashModel:
    r"""Fitting the SPLASH model.

    The SplashModel class calculates the predictions of the SPLASH v1.0 model
    :cite:p:`davis:2017a`. The input variables of latitude, elevation, temperature,
    precipitation and sunshine fraction of observations are initially used to calculate
    solar and evaporative fluxes, which are stored in the ``solar`` and ``evap``
    attributes as instances of :class:`~pyrealm.splash.solar.DailySolarFluxes` and
    :class:`~pyrealm.splash.evap.DailyEvapFluxes`.

    The inputs to a SplashModel are expected to be numpy arrays with time varying along
    the first dimension. Other dimensions represent observations at sites on a
    particular date.  The ``dates`` argument is expected to be a Calendar object with
    the same length as the first dimension.

    The main use of the SplashModel object is then to calculate the expected actual
    evapotranspiration (AET), soil moisture and runoff across the time series:

    * The :meth:`~pyrealm.splash.splash.SplashModel.calculate_soil_moisture` returns
      these calculations, given an initial estimate of soil moisture in observed sites.
      This method simply iterates over the days, applying the
      :meth:`~pyrealm.splash.splash.SplashModel.estimate_daily_water_balance` method to
      calculate the daily water balance, given the soil moisture of the preceeding day.

    * The :meth:`~pyrealm.splash.splash.SplashModel.estimate_initial_soil_moisture`
      method can be used to estimate an initial soil moisture for a time series from the
      first year of data in a time series.

    Args:
        lat: The latitude of observations
        elv: The elevation of observations (m), also used to calculate atmospheric
            pressure.
        sf: The sunshine fraction (0-1, unitless)
        tc: Air temperature (°C)
        pn: Precipitation (mm/day)
        dates: The dates of the time series
        kWm: The maximum soil moisture capacity, defaulting to 150 (mm)
    """

    def __init__(
        self,
        lat: NDArray[np.float64],
        elv: NDArray[np.float64],
        sf: NDArray[np.float64],
        tc: NDArray[np.float64],
        pn: NDArray[np.float64],
        dates: Calendar,
        kWm: NDArray[np.float64] = np.array([150.0]),
        core_const: CoreConst = CoreConst(),
        bounds_checker: BoundsChecker = BoundsChecker(),
    ):
        # Check input sizes are congurent
        # TODO - xarray would be good here for identifying axes and
        #        checking congruence more widely.
        self.shape: tuple = check_input_shapes(elv, lat, sf, tc, pn)
        """The array shape of the input variables"""

        if self.shape[0] == 1:
            self.shape = (len(dates), *self.shape[1:])
        elif self.shape[0] != len(dates):
            raise ValueError(
                "The first dimension of inputs must either match the number of dates or"
                " have a length of one."
            )

        # Broadcast all the inputs over time to simplify the daily indexing if any
        # inputs are constant over time
        elv = broadcast_time(elv, self.shape)
        lat = broadcast_time(lat, self.shape)
        sf = broadcast_time(sf, self.shape)
        tc = broadcast_time(tc, self.shape)
        pn = broadcast_time(pn, self.shape)

        self.elv: NDArray[np.float64] = elv
        """The elevation of sites."""
        self.lat: NDArray[np.float64] = bounds_checker.check("lat", lat)
        """The latitude of sites."""
        self.sf: NDArray[np.float64] = bounds_checker.check("sf", sf)
        """The sunshine fraction (0-1) of daily observations."""
        self.tc: NDArray[np.float64] = bounds_checker.check("tc", tc)
        """The air temperature in °C of daily observations."""
        self.pn: NDArray[np.float64] = bounds_checker.check("pn", pn)
        """The precipitation in mm of daily observations."""
        self.dates: Calendar = dates
        """The dates of observations along the first array axis."""
        self.kWm: NDArray[np.float64] = bounds_checker.check("kWm", kWm)
        """The maximum soil water capacity for sites."""

        # TODO - potentially allow _actual_ climatic pressure data as an input
        self.pa: NDArray[np.float64] = calc_patm(elv, core_const=core_const)
        """The atmospheric pressure at sites, derived from elevation"""

        # Calculate the daily solar fluxes - these are invariant across the simulation
        self.solar: DailySolarFluxes = DailySolarFluxes(
            latitude=lat,
            elevation=elv,
            dates=dates,
            sunshine_fraction=sf,
            temperature=tc,
        )
        """Estimated solar fluxes for observations"""

        # Initialise the evaporative flux class
        self.evap: DailyEvapFluxes = DailyEvapFluxes(
            solar=self.solar, pa=self.pa, tc=tc, core_const=core_const
        )
        """Estimated evaporative fluxes for observations"""

    def estimate_initial_soil_moisture(
        self,
        wn_init: NDArray[np.float64] | None = None,
        max_iter: int = 10,
        max_diff: float = 1.0,
        return_convergence: bool = False,
        verbose: bool = False,
    ) -> NDArray[np.float64]:
        """Estimate initial soil moisture.

        This method uses the first year of data provided to a SplashModel instance to
        estimate initial values for the soil moisture data. The process assumes that the
        soil moisture cycle is stationary over the first year of data and iteratively
        updates an initial guess at start of year soil moisture until those values are
        within a given tolerance of the end of year values. The method cannot be run
        when less than one year of data is provided to the model.

        The user can provide an array of initial values across sites, defaulting to an
        initial guess of zero soil moisture in all sites . The user can also control the
        maximum number of update iterations and the accepted tolerance for convergence.
        The method will normally fail if the estimates do not converge, but the
        ``return_convergence`` option can be used to return the estimated soil moisture
        at each iteration regardless of the success of convergence.

        Args:
            wn_init: An optional estimate of the start of year soil moisture.
            max_iter: The maximum number of iterations used to achieve convergence.
            max_diff: The maximum acceptable difference between year start and year end
                soil moisture,
            return_convergence: Optionally return an array of soil moistures at the end
                of each iteration even when convergence fails.
            verbose: Optionally turn on detailed logging of the iteration process.

        Returns:
            An array of the estimated starting soil moisture. If ``return_convergence``
            is set to True, the returned array will have an additional dimension for
            each iteration of the equilibration loop.

        Raises:
            ValueError: The input data are of the wrong shape, contain invalid values or
                do not include at least a full year of data.
            RuntimeError: The estimation fails to converge within the set number of
                iterations.
        """

        # Initialise loop termination
        equilibrated = False
        n_iter = 0
        wn_ret = []

        if wn_init is not None:
            # Check the shape is the same as the shape of a slice along axis 0
            if wn_init.shape != self.shape[1:]:
                raise ValueError("Incorrect shape in wn_init")
            if np.any((wn_init < 0) | (wn_init > self.kWm)):
                raise ValueError("Soil moisture must be between 0 and kWm")
            wn_start = wn_init
        else:
            wn_start = np.zeros(self.shape[1:])

        # Find a date one year into the future from the first calendar date.
        date_start = pd.Timestamp(self.dates[0].date)
        date_end = date_start + pd.DateOffset(years=1)
        num_days = (date_end - date_start).days

        if len(self.dates) < num_days:
            raise ValueError("Cannot equilibrate - less than one year of data")

        # Run the equilibration loop
        while (not equilibrated) and (n_iter < max_iter):
            # Track the iterations
            n_iter += 1

            # Loop over the calendar object, updating the soil_moisture array
            wn_day = wn_start
            for day_idx in range(num_days):
                # Calculate aet, soil moisture and runoff:
                _, wn_day, _ = self.estimate_daily_water_balance(
                    previous_wn=wn_day, day_idx=day_idx
                )

            # Calculate the difference between the start of year soil moisture and the
            # final day of the year and then update the start point to the end of year.
            diff_sm = np.abs(wn_start - wn_day)
            cur_diff = np.nanmax(diff_sm)
            wn_start = wn_day

            # Optionally store the soil moisture at the end of each loop
            if return_convergence:
                wn_ret.append(wn_start)

            # Report if verbose
            if verbose:
                print(f"Iteration: {n_iter}; maximum difference: {cur_diff}")

            if cur_diff <= max_diff:
                equilibrated = True

        # Check for convergence failure before returning the final values.
        if not equilibrated:
            msg = (
                f"Initial soil moisture did not converge within {n_iter} iterations:"
                f"maximum absolute difference = {cur_diff}"
            )
            if return_convergence:  # always returns without raising an error
                warnings.warn(msg)
            else:
                raise RuntimeError(msg)

        if return_convergence:  # returns the values of wn at each iteration
            return np.array(wn_ret)  # (n_iter, *wn_start.shape)
        else:
            return wn_start

    def estimate_daily_water_balance(
        self, previous_wn: NDArray[np.float64], day_idx: int | None = None
    ) -> tuple[NDArray, NDArray, NDArray]:
        r"""Estimate the daily water balance.

        This function estimates the daily water balance within observations. The
        function first calculates the expected actual evapotranspiration (mm d-1,
        :math:`\textrm{AET}_{[t]}`), given the soil moisture from the preceeding day
        (mm, :math:`W_{n[t-1]}`). Those are then used, along with the precipitation (mm
        d-1, :math:`P_{[t]}`) and condensation (mm d-1, :math:`C_{[t]}`) for the current
        day, to calculate the current soil moisture (mm, :math:`W_{n[t]}`) as:

        .. math::

            W_{n[t]} = W_{n[t-1]} + P_{[t]} + C_{[t]} - \textrm{AET}_{[t]}.

        When the resulting soil moisture exceeds the maximum capacity of the soil
        (``kWm``), the excess is allocated to run off, leaving the soil saturated. Note
        that the soil moisture is not altered by subsurface flow: there is not vertical
        or horizontal transfer of water from the soil, only losses through
        evapotranspiration. Negative soil moisture values are replaced by zero.

        By default, ``previous_wn`` is expected to provide estimates for all
        observations across all days in the model, but ``day_idx`` can be set to provide
        an estimate for only one particular day, for use in iterating over time series.

        Args:
            day_idx: Optionally, the index of the date for which to calculate water
                balance.
            previous_wn: Soil moisture estimates for the preceeding day (mm)

        Returns:
            A tuple of numpy arrays containing estimated  AET, daily soil moisture and
            runoff.
        """

        # Check day_idx inputs to map either the single time index given in day_idx or
        # the whole dataset.
        if day_idx is None:
            splash_shape = self.shape
            didx: int | slice = slice(self.shape[0])
        else:
            splash_shape = self.shape[1:]
            didx = day_idx
        try:
            check_input_shapes(previous_wn, shape=splash_shape)
        except ValueError:
            msg = (
                "The shape of previous_wn does not match the existing SPLASH model data"
            )
            raise ValueError(msg)

        # Calculate the expected aet_d given the previous wn
        if np.any((previous_wn < 0) | (previous_wn > self.kWm)):
            raise ValueError("Soil moisture must be between 0 and kWm")

        aet = self.evap.estimate_aet(wn=previous_wn, day_idx=day_idx)

        # Calculate current soil moisture, mm
        current_wn = previous_wn + self.pn[didx] + self.evap.cond[didx] - aet

        # Partition current_wn into soil moisture and runoff (ro), mm
        # - allocate excess sm to runoff and clip out negative sm
        ro = np.clip(current_wn, self.kWm, np.inf) - self.kWm
        wn = np.clip(current_wn, 0, self.kWm)

        # Return values, ignoring the type clash that estimate_aet _can_ return
        # additional arrays. aet here is explicitly a single array not a tuple.
        return aet, wn, ro  # type: ignore

    def calculate_soil_moisture(
        self,
        wn_init: NDArray[np.float64],
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Calculate the soil moisture, AET and runoff from a SplashModel.

        This function takes an initial array of soil moisture values for the first
        observations in a SplashModel time series and then iteratively applies the daily
        water balance calculations along the time axis using the
        :meth:`~pyrealm.splash.splash.SplashModel.estimate_daily_water_balance` method.
        This produces the expected actual evapotranspiration (AET), soil moisture,
        runoff and for all sites across the time series.

        Args:
            wn_init: The initial state of the soil moisture for observations

        Returns:
            A tuple of numpy arrays containing predicted AET, soil moisture and runoff.
        """

        try:
            check_input_shapes(wn_init, shape=self.shape[1:])
        except ValueError:
            msg = "The shape of wn_init does not match the existing SPLASH model data"
            raise ValueError(msg)

        # Create storage for outputs
        aet_out = np.full_like(self.tc, np.nan)
        wn_out = np.full_like(self.tc, np.nan)
        ro_out = np.full_like(self.tc, np.nan)

        if np.any((wn_init < 0) | (wn_init > self.kWm)):
            raise ValueError("Soil moisture must be between 0 and kWm")

        curr_wn = wn_init

        for day_idx in range(self.shape[0]):
            # Calculate the balance for this date, updating the input for
            # the following day
            aet, curr_wn, ro = self.estimate_daily_water_balance(
                curr_wn, day_idx=day_idx
            )

            # Store the outputs to return
            aet_out[day_idx] = aet
            wn_out[day_idx] = curr_wn
            ro_out[day_idx] = ro

        return aet_out, wn_out, ro_out
