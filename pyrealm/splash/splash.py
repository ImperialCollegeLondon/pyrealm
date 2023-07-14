"""The splash submodule provides the main SplashModel class for calculating predictions
under the SPLASH model.
"""  # noqa: D205, D415

from typing import Optional, Union

import numpy as np

from pyrealm.splash.const import kWm
from pyrealm.splash.evap import DailyEvapFluxes, elv2pres
from pyrealm.splash.logger import logger
from pyrealm.splash.solar import DailySolarFluxes
from pyrealm.splash.utilities import Calendar
from pyrealm.utilities import check_input_shapes


class SplashModel:
    """Fitting the SPLASH model.

    The SplashModel class calculates the predictions of the SPLASH v1.0 model
    :cite:p:`davis:2017a`. The input variables of latitude, elevation, temperature,
    precipitation and sunshine fraction of observations are initially used to calculate
    solar and evaporative fluxes, which are stored in the ``solar`` and ``evap``
    attributes as instances of :class:`~pyrealm.splash.solar.DailySolarFluxes` and
    :class:`~pyrealm.splash.evap.DailyEvapFluxes`.



    From those initial calculations, the SPLASH model can then apply a simple
    calculation to iteratively track soil moisture and other evaporative fluxes through
    time.

        wn[t] = wn[t-1] + self.pn[t] + self.evap.cond[t] - self.evap.aet_d[t]

    Where the resulting soil moisture exceeds the maximum capacity of the soil (kWm),
    the excess is allocated to run off, leaving the soil saturated. The
    :meth:`~pyrealm.splash.splash.SplashModel.calc_splash_daily` method is used to
    calculate soil moisture and runoff for a given day, given the soil moisture of the
    preceeding day. The
    meth:`~pyrealm.splash.splash.SplashModel.equilibrate_soil_moisture` method can be
    used to estimate an initial soil moisture for a time series.

    The inputs to SplashModel are expected to be  numpy arrays with time varying along
    the first dimension. The dates argument is expected to be a Calendar object with the
    same length as the first dimension.

    Args:
        lat: The latitude of observations
        elv: The elevation of observations, also used to calculate atmospheric pressure.
        sf: The sunshine fraction (0-1, unitless)
        tc: Air temperature (°C)
        pn: Precipitation (mm/day)
        dates: The dates of the time series
        kWm: The maximum soil moisture capacity, defaulting to 150 (mm)
    """

    def __init__(
        self,
        lat: np.ndarray,
        elv: np.ndarray,
        sf: np.ndarray,
        tc: np.ndarray,
        pn: np.ndarray,
        dates: Calendar,
        kWm: np.ndarray = np.array([150.0]),
    ):
        # TODO - check input sizes are congurent and maybe think about broadcasting lat
        #        and elv. xarray would be good here.

        # Assign required public attributes
        self.elv = elv
        """The elevation of sites."""
        self.lat = lat
        """The latitude of sites."""
        self.sf = sf
        """The sunshine fraction (0-1) of daily observations."""
        self.tc = tc
        """The air temperature in °C of daily observations."""
        self.pn = pn
        """The precipitation in mm of daily observations."""
        self.dates = dates
        """The dates of observations along the first array axis."""
        self.kWm = kWm
        """The maximum soil water capacity for sites."""

        # TODO - check and swap in pyrealm function - noting that this uses 15°C as the
        #        standard atmosphere, where pyrealm _currently_ uses 25°C for no good
        #        reason.
        #      - potentially allow _actual_ climatic pressure data.
        self.pa = elv2pres(elv)
        """The atmospheric pressure at sites, derived from elevation"""

        # Calculate the daily solar fluxes - these are invariant across the simulation
        self.solar: DailySolarFluxes = DailySolarFluxes(
            lat=lat, elv=elv, dates=dates, sf=sf, tc=tc
        )
        """Estimated solar fluxes for observations"""

        # Initialise the evaporative flux class
        self.evap: DailyEvapFluxes = DailyEvapFluxes(self.solar, pa=self.pa, tc=tc)
        """Estimated evaporative fluxes for observations"""

    def estimate_initial_soil_moisture(
        self,
        wn_init: Optional[np.ndarray] = None,
        max_iter: int = 10,
        max_diff: float = 1.0,
        verbose: bool = True,
    ) -> np.ndarray:
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

        Args:
            wn_init: An optional estimate of the start of year soil moisture.
            max_iter: The maximum number of iterations used to achieve convergence.
            max_diff: The maximum acceptable difference between year start and year end
                soil moisture,
            verbose: Optionally turn on detailed logging of the iteration process.

        Returns:
            An array of the estimated starting soil moisture.
        """

        # Initialise loop termination
        equilibrated = False
        n_iter = 0
        diff_sm = np.array([np.nan])

        if wn_init is not None:
            # Check the shape is the same as the shape of a slice along axis 0
            if wn_init.shape != self.tc[0].shape:
                raise ValueError("Incorrect shape in wn_init")
            wn_start = wn_init
        else:
            wn_start = np.zeros_like(self.tc[0])

        # Find a date one year into the future from the first calendar date.
        # TODO - fix leap year handling and non Jan 1 starts

        if self.tc.shape[0] < 365:
            raise ValueError("Cannot equilibrate - less than one year of data")

        # Run the equilibration loop
        while (not equilibrated) and (n_iter < max_iter):
            # Track the iterations
            n_iter += 1

            # Loop over the calendar object, updating the soil_moisture array
            wn_day = wn_start
            for day_idx in np.arange(366):
                # Calculate aet, soil moisture and runoff:
                _, wn_day, _ = self.estimate_daily_water_balance(
                    previous_wn=wn_day, day_idx=day_idx
                )

            # Calculate the difference between the start of year soil moisture and the
            # final day of the year and then update the start point to the end of year.
            diff_sm = np.abs(wn_start - wn_day)
            wn_start = wn_day

            # Report if verbose
            if verbose:
                logger.info(f"Iteration: {n_iter}; maximum difference: {diff_sm.max()}")

            if np.nanmax(diff_sm) <= max_diff:
                equilibrated = True

        # Check for convergence failure.
        if not equilibrated:
            raise RuntimeError(
                f"Initial soil moisture did not converge within {n_iter} iterations:"
                f"maximum absolute difference = {diff_sm.max()}"
            )

        return wn_start

    def estimate_daily_water_balance(
        self, previous_wn: np.ndarray, day_idx: Optional[int] = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Estimate the daily water balance.

        The daily soil moisture (wn) is estimated using the soil moisture from the
        preceeding day, the precipitation for each day (``pn``) and the calculated
        condensation (``cn``) and AET given the evaporative fluxes:

        ..math::

            \textrm{wn}_{t} = \textrm{wn}_{t-1} + \textrm{pn} +
                              \textrm{cn} - \textrm{aet}

        The calculated soil moisture is then partitioned into soil moisture and any
        surplus runoff, given the maximum soil capacity for each observation. Negative
        soil moisture values are replaced by zero.

        By default, ``previous_wn`` is expected to provide estimates for all
        observations across all days in the model, but ``day_idx`` can be set to provide
        an estimate for only one particular day, for use in iterating over time series.

        Args:
            day_idx: Optionally, the index of the date for which to calculate water
                balance.
            previous_wn: Soil moisture estimates for the preceeding day (mm)

        Returns:
            A tuple of numpy arrays containing predicted AET, daily soil moisture and
            runoff.
        """

        # Check day_idx inputs to map either the single time index given in day_idx or
        # the whole dataset.
        if day_idx is None:
            check_input_shapes(previous_wn, self.pn)
            didx: Union[int, slice] = slice(self.pn.shape[0])
        else:
            check_input_shapes(previous_wn, self.pn[day_idx])
            didx = day_idx

        # Calculate the expected aet_d given the previous wn
        aet = self.evap.estimate_aet(wn=previous_wn, day_idx=day_idx)

        # Calculate current soil moisture, mm
        current_wn = previous_wn + self.pn[didx] + self.evap.cond[didx] - aet

        # Partition current_wn into soil moisture and runoff (ro), mm
        # - allocate excess sm to runoff and clip out negative sm
        ro = np.clip(current_wn, kWm, np.inf) - kWm
        wn = np.clip(current_wn, 0, kWm)

        # Return values, ignoring the type clash that estimate_aet _can_ return
        # additional arrays. aet here is explicitly a single array not a tuple.
        return aet, wn, ro  # type: ignore

    def iterate_water_balance(
        self,
        wn_init: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Iteratively apply daily water balance calculations along time axis.

        Args:
            wn_init: The initial state of the soil moisture for observations

        Returns:
            A tuple of numpy arrays containing predicted AET, soil moisture and runoff.

        """

        # TODO - check input shapes

        # Create storage for outputs
        aet_out = np.full_like(self.tc, np.nan)
        wn_out = np.full_like(self.tc, np.nan)
        ro_out = np.full_like(self.tc, np.nan)

        curr_wn = wn_init
        for day_idx in np.arange(self.pn.shape[0]):
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
