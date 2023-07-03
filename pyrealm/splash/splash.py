"""The splash submodule provides the main class for calculating predictions under the
SPLASH model.
"""
from typing import Optional

import numpy as np

from pyrealm.splash.const import kCw, kWm
from pyrealm.splash.evap import DailyEvapFluxes, elv2pres
from pyrealm.splash.logger import logger
from pyrealm.splash.solar import DailySolarFluxes
from pyrealm.splash.utilities import Calendar


class SplashModel:
    """Create a SPLASH model instance.

    The SplashModel class provides an interface to calculate the predictions of SPLASH
    v1 [TODO - citation]. When a model instance is created, the input variables are used
    to calculate all over the predicted radiative and evaporative fluxes, except for the
    estimation of daily AET.

    The instance can be used to estimate the initial soil moisture content (wn): given a
    year's worth of data, the
    """

    def __init__(
        self,
        lat: np.ndarray,
        elv: np.ndarray,
        dates: Calendar,
        sf: np.ndarray,
        tc: np.ndarray,
        pn: np.ndarray,
        kWm: np.ndarray = np.array([150.0]),
    ):
        # TODO - check input sizes are congurent
        # Error handle and assign required public variables:
        self.elv = elv
        self.lat = lat
        self.sf = sf
        self.tc = tc
        self.pn = pn
        self.dates = dates
        self.kWm = kWm

        # TODO - check and swap in pyrealm function
        self.pa = elv2pres(elv)

        # Calculate the daily solar fluxes - these are invariant across the simulation
        self.solar: DailySolarFluxes = DailySolarFluxes(
            lat=lat, elv=elv, dates=dates, sf=sf, tc=tc
        )

        # Initialise the evaporative flux class
        self.evap: DailyEvapFluxes = DailyEvapFluxes(self.solar, pa=self.pa, tc=tc)

    def equilibrate_soil_moisture(
        self,
        verbose: bool = True,
        wn_init: Optional[np.ndarray] = None,
        max_iter: int = 10,
        max_diff_sm: float = 1.0,
    ):
        """
        Name:     SPLASH.spin
        Input:    - DATA class, (d)
                  - bool, (to_write)
        Output:   None.
        Features: Spins up the daily soil moisture, creating a daily soil
                  moisture vector (wn_vec) and previous day's soil moisture
                  value (wn).
        Depends:  quick_run
        """

        # Initialise loop termination
        equilibrated = False
        n_iter = 0

        if wn_init is not None:
            # Check the shape is the same as the shape of a slice along axis 0
            if wn_init.shape != self.tc[0].shape:
                raise ValueError("Incorrect shape in wn_init")
        else:
            wn_init = np.full_like(self.tc[0], np.nan)

        # Find a date one year into the future from the first calendar date.
        # TODO - fix leap year handling and non Jan 1 starts

        if self.tc.shape[0] < 365:
            raise ValueError("Cannot equilibrate - less than one year of data")

        # Run the equilibration loop
        while (not equilibrated) or (n_iter <= max_iter):
            # Track the iterations
            n_iter += 1

            # Loop over the calendar object, updating the soil_moisture array
            wn_day = wn_init
            for cday in np.arange(367):
                # Calculate soil moisture and runoff:
                wn_day, _ = self.calc_splash_daily(
                    date=self.dates[cday],
                    wn=wn_day,
                )

            # Calculate the change in the starting soil moisture and update the initial
            # guess to the end of year value
            diff_sm = np.abs(wn_init - wn_day)
            wn_init = wn_day

            # Report if verbose
            if verbose:
                logger.info(
                    f"Iter: {n_iter}; maximum soil moisture differential: {diff_sm.max()}"
                )

            if np.all(diff_sm <= max_diff_sm):
                equilibrated = True

        return wn_day

    def calc_splash_daily(
        self,
        wn: np.ndarray,
    ) -> tuple[DailyEvapFluxes, np.ndarray, np.ndarray]:
        """
        Name:     SPLASH.run_one_day
        Inputs:   - int, day of year (n)
                  - int, year (y)
                  - float, soil water content, mm (wn)
                  - float, fraction of bright sunshine (sf)
                  - float, air temperature, deg C (tc)
                  - float, precipitation, mm (pn)
        Outputs:  None
        Features: Runs SPLASH model for one day.
        Depends:  - kCw
                  - kWm
                  - EVAP
        """

        # Calculate evaporative supply rate (sw), mm/h
        # TODO - spatial variaion in kWm
        sw = kCw * wn / self.kWm

        # Calculate the expected aet_d given the provided wn
        self.evap.estimate_hi_and_aet(sw=sw)

        # Calculate soil moisture (sm), mm
        sm = wn + self.pn + self.evap.cond - self.evap.aet_d

        # Calculate runoff (ro), mm
        # - allocate excess sm to runoff and clip out negative sm
        ro = np.clip(sm, kWm, np.inf) - kWm
        sm = np.clip(sm, 0, kWm)

        return sm, ro
