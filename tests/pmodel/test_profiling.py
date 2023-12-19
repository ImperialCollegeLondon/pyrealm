"""Profiling test class.

Runs a profiler on the pmodel implementation to identify runtime
bottlenecks.
"""

from importlib import resources

import numpy as np
import pytest
import xarray

from pyrealm.pmodel.pmodel_environment import PModelEnvironment


@pytest.mark.profiling
class TestClass:
    """Test class for the profiler running on the pmodel implementation."""

    @pytest.fixture()
    def setup(self):
        """Setting up the pmodel and loading the test data set."""
        # Loading the dataset:
        dpath = resources.files("pyrealm_build_data") / "inputs_data_24.25.nc"

        ds = xarray.load_dataset(dpath)

        # TODO - this is a bit of a hack because of the current unfriendly handling of
        #  data that does not form neat blocks of daily data in the subdaily module. The
        #        test data is a slice along longitude 24.25°E (Finland --> Crete -->
        #        Botswana) so the actual local times needed for the subdaily module are
        #        offset from the UTC times in the data. This step reduces the input data
        #        to complete daily blocks of data using local time

        ds = ds.sel(time=slice("2000-01-01T01:59", "2001-12-31T01:59"))
        ds['local_time_offset'] = (ds["lon"] // 15 * 3.6e12).astype("timedelta64[ns]")
        ds['local_time'] = ds["time"] - ds['local_time_offset']

        # Variable set up
        # Air temperature in Kelvin
        self.tc = (ds["temp"]).to_numpy()
        # Atmospheric pressure in Pascals
        self.patm = ds["patm"].to_numpy()
        # Obtain VPD and remove negative values
        self.vpd = ds["vpd"].to_numpy()
        self.vpd = np.clip(self.vpd, 0, np.inf)
        # Extract fAPAR (unitless)
        self.fapar = ds["fapar"].to_numpy()
        # Gather PPFD µmole/m2/s1
        self.ppfd = ds["ppfd"].to_numpy()
        # Define atmospheric CO2 concentration (ppm)
        self.co2 = np.ones_like(self.tc) * 400
        # Define the local time at different longitudes
        self.local_time = ds["local_time"].to_numpy().squeeze()

        # Generate and check the PModelEnvironment
        self.pm_env = PModelEnvironment(
            tc=self.tc, patm=self.patm, vpd=self.vpd, co2=self.co2
        )

    def test_profiling_example(self, setup):
        """Running the profiler on the pmodel."""
        from pyrealm.pmodel import C3C4Competition, CalcCarbonIsotopes, PModel

        # Profiling the PModel submodule
        # Standard C3 PModel
        pmod_c3 = PModel(env=self.pm_env, kphio=1 / 8)
        pmod_c3.estimate_productivity(fapar=self.fapar, ppfd=self.ppfd)
        pmod_c3.summarize()

        # Standard C4 PModel
        pmod_c4 = PModel(env=self.pm_env, kphio=1 / 8, method_optchi="c4")
        pmod_c4.estimate_productivity(fapar=self.fapar, ppfd=self.ppfd)
        pmod_c4.summarize()

        # Profiling the Competition submodule
        # Competition, using annual GPP from µgC m2 s to g m2 yr
        gpp_c3_annual = pmod_c3.gpp * (60 * 60 * 24 * 365) * 1e-6
        gpp_c4_annual = pmod_c4.gpp * (60 * 60 * 24 * 365) * 1e-6

        # Fit the competition model - making some extrenely poor judgements about what
        # is cropland and what is below the minimum temperature that really should be
        # fixed.
        comp = C3C4Competition(
            gpp_c3=gpp_c3_annual,
            gpp_c4=gpp_c4_annual,
            treecover=np.array([0.5]),
            below_t_min=np.full_like(self.pm_env.tc, False, dtype="bool"),
            cropland=np.full_like(self.pm_env.tc, False, dtype="bool"),
        )

        comp.summarize()

        # Profiling the isotopes submodule
        # Create some entirely constant atmospheric isotope ratios
        constant_d13CO2 = np.array([-8.4])
        constant_D14CO2 = np.array([19.2])

        # Calculate for the C3 model
        isotope_c3 = CalcCarbonIsotopes(
            pmod_c3, d13CO2=constant_d13CO2, D14CO2=constant_D14CO2
        )
        isotope_c3.summarize()

        # Calculate for the C4 model
        isotope_c4 = CalcCarbonIsotopes(
            pmod_c4, d13CO2=constant_d13CO2, D14CO2=constant_D14CO2
        )
        isotope_c4.summarize()

        # Calculate the expected isotopic patterns in locations given the competition
        # model
        comp.estimate_isotopic_discrimination(
            d13CO2=constant_d13CO2,
            Delta13C_C3_alone=isotope_c3.Delta13C,
            Delta13C_C4_alone=isotope_c4.Delta13C,
        )

        comp.summarize()

    def test_profile_subdaily(self, setup):
        """Profiling the subdaily submodule."""
        from pyrealm.pmodel import FastSlowPModel, FastSlowScaler

        # FastSlowPModel with 1 hour noon acclimation window
        # TODO - make the code applicable to a dataset with multiple longitudes
        fsscaler = FastSlowScaler(self.local_time)
        fsscaler.set_window(
            window_center=np.timedelta64(12, "h"),
            half_width=np.timedelta64(1, "h"),
        )
        fs_pmod = FastSlowPModel(
            env=self.pm_env,
            fs_scaler=fsscaler,
            handle_nan=True,
            fapar=self.fapar,
            ppfd=self.ppfd,
            alpha=1 / 15,
        )
        return fs_pmod
