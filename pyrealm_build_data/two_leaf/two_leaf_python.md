---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: pyrealm_python3
  language: python
  name: pyrealm_python3
---

# Two leaf radiative transfer model

This notebook is a draft implementation of the de Pury & Farquhar (1997) two leaf, two
stream model for use with `pyrealm`. This model is chosen to provide a better
representation than the big leaf model and to align closely to the workings of the BESS
model (Ryu et al. 2011)

```{code-cell} ipython3
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pvlib.location import Location

from pyrealm.pmodel import PModelEnvironment, PModel, SubdailyPModel, SubdailyScaler
```

## Test data

We run this trial for the Vielsalm FLUXNET site in Belgium (details appear in Table 2
Mengoli et al.). The data is loaded from a file that combines a range of sources (see
`BE-Vie_merge.py`).

```{code-cell} ipython3
forcing_data = pd.read_csv('merged_BE-Vie_data.csv')
forcing_data["time"] = pd.to_datetime(forcing_data["time"])
forcing_data
```

## Solar position

The two leaf, two stream model uses the solar elevation to better estimate the behaviour
of light within the sun and shade partitions of the canopy.

```{code-cell} ipython3
# Calculate the solar elevation for the observations
pv_location = Location(
    latitude=50.30493, longitude=5.99812, tz="Etc/GMT+1", altitude=493, name="BE-Vie"
)
solar_position = pv_location.get_solarposition(forcing_data["time"])
solar_position
```

The resulting solar elevation looks correct.

```{code-cell} ipython3
week_one = solar_position[
    (solar_position.index > "2014-08-01") & (solar_position.index < "2014-08-08")
]

fig, ax = plt.subplots(ncols=1, nrows=1)
week_one.plot(y="elevation", ax=ax, legend=False)
ax.axhline(0, color="darkgray")
ax.set_xlabel("Local time")
ax.set_ylabel("(degrees)");
```

```{code-cell} ipython3
# Add solar elevation as radians to the forcing dataframe
forcing_data["solar_zenith"] = np.deg2rad(solar_position["zenith"].to_numpy())
forcing_data["solar_elevation"] = np.deg2rad(solar_position["elevation"].to_numpy())
```

```{code-cell} ipython3
week_one = forcing_data[
    (solar_position.index > "2014-08-01") & (solar_position.index < "2014-08-08")
]

fig, ax = plt.subplots(ncols=1, nrows=1)
week_one.plot(y="solar_elevation", ax=ax, legend=False)
ax.axhline(0, color="darkgray")
ax.set_xlabel("Local time")
ax.set_ylabel("(radians)");
```

## Radiance partitioning

The next step is to calculate radiance partitioning to give the sun and shade irradiance
and beam extinction coefficient.

The function below implements the calculations from deP&F, taking solar elevation
($\beta$, radians), PPFD, LAI and atmospheric pressure along with parameterisation from
tables 2 and 3 in deP&F:

* `fa`: scattering coefficient of PAR in the atmosphere
* `sigma`: leaf scattering coefficient of PAR (relections and transmissivity)
* `rho_cd`: canopy reflection coefficient for diffuse PAR
* `kd_prime`: diffuse and scattered diffuse PAR extinction coefficient

```{code-cell} ipython3
class TwoLeafIrradiance:
    def __init__(
        self,
        beta_angle,
        PPFD,
        LAI,
        PATM,
        solar_obscurity_angle=np.deg2rad(1),
        PA0=101325,
        fa=0.426,
        sigma=0.15,
        rho_cd=0.036,
        kd_prime=0.719,
    ):
        """Partition solar irradiance

        The references in parentheses e.g. (A23) refer to the equation sequence in
        deP&F 1997 and recreated here.
        """

        # Store key inputs
        self.solar_obscurity_angle = solar_obscurity_angle
        self.beta_angle = beta_angle

        # beam extinction coefficient, clipped below solar_obscurity_angle
        self.kb = np.where(
            beta_angle > solar_obscurity_angle, 0.5 / np.sin(self.beta_angle), 30
        )

        # beam and scattered-beam extinction coefficient
        kb_prime = np.where(
            beta_angle > solar_obscurity_angle, 0.46 / np.sin(self.beta_angle), 27
        )

        # fraction of diffuse radiation (A23 & A25)
        m = (PATM / PA0) / np.sin(beta_angle)
        fd = (1 - 0.72**m) / (1 + (0.72**m * (1 / fa - 1)))

        # beam irradiance horizontal leaves (A20)
        rho_h = (1 - np.sqrt(1 - sigma)) / (1 + np.sqrt(1 - sigma))

        # beam irradiance uniform leaf-angle distribution (A19)
        rho_cb = 1 - np.exp(-2 * rho_h * self.kb / (1 + self.kb))

        # diffuse radiation, truncating at zero
        I_d = np.clip(PPFD * fd, a_min=0, a_max=np.inf)

        # beam irradiance
        I_b = PPFD * (1 - fd)

        # scattered beam irradiance (A8)
        self.I_bs = I_b * (1 - rho_cb) * kb_prime * np.exp(-kb_prime * LAI) - (
            1 - sigma
        ) * self.kb * np.exp(-self.kb * LAI)

        # Irradiance absorbed by the whole canopy (Eqn 13)
        self.I_c = (1 - rho_cb) * I_b * (1 - np.exp(-kb_prime * LAI)) + (
            1 - rho_cd
        ) * I_d * (1 - np.exp(-kb_prime * LAI))

        # for the sunlit leaves, the components are (Eqn 20):

        self.Isun_beam = I_b * (1 - sigma) * (1 - np.exp(-self.kb * LAI))
        self.Isun_diffuse = (
            I_d
            * (1 - rho_cd)
            * (1 - np.exp(-(kd_prime + self.kb) * LAI))
            * kd_prime
            / (kd_prime + self.kb)
        )
        self.Isun_scattered = I_b * (
            (
                (1 - rho_cb)
                * (1 - np.exp(-(kb_prime + self.kb) * LAI))
                * kb_prime
                / (kb_prime + self.kb)
            )
            - ((1 - sigma) * (1 - np.exp(-2 * self.kb * LAI)) / 2)
        )

        # Irradiance absorbed by the sunlit fraction of the canopy (Eqn 20):
        self.I_csun = self.Isun_beam + self.Isun_diffuse + self.Isun_scattered

        # Irradiance absorbed by the shaded fraction of the canopy (Eqn 21):
        # and including a clause to exclude the hours of obscurity
        self.I_cshade = np.where(
            beta_angle > solar_obscurity_angle, self.I_c - self.I_csun, 0
        )
```

We can now calculate the radiance partitioning for the site:

```{code-cell} ipython3
two_leaf_irrad = TwoLeafIrradiance(
    beta_angle=forcing_data["solar_elevation"].to_numpy(),
    PPFD=forcing_data["ppfd"].to_numpy(),
    LAI=forcing_data["LAI"].to_numpy(),
    PATM=forcing_data["patm"].to_numpy(),
)
```

Keith's original R demo generated the following values for `2014-08-01 12:30:00`:

`[580.2101296936461, 55.876940533221614, 0.9821211680903654, 0.6011949063494533]`

We can validate that against the calculated arrays.

```{code-cell} ipython3
idx = np.where(forcing_data["time"].to_numpy() == np.datetime64("2014-08-01 12:30:00"))

test_vals = [
    val_array[idx][0]
    for val_array in [
        two_leaf_irrad.I_csun,
        two_leaf_irrad.I_cshade,
        two_leaf_irrad.beta_angle,
        two_leaf_irrad.kb,
    ]
]

print(test_vals)
```

## Big Leaf P Model estimates

We can now fit the current "big leaf" implementations using the standard and subdaily approaches.

```{code-cell} ipython3
pmod_env = PModelEnvironment(
    tc=forcing_data["tc"].to_numpy(),
    patm=forcing_data["patm"].to_numpy(),
    co2=forcing_data["co2"].to_numpy(),
    vpd=forcing_data["vpd"].to_numpy(),
)

# Standard P Model
standard_pmod = PModel(pmod_env, kphio=1 / 8)
standard_pmod.estimate_productivity(
    fapar=forcing_data["fapar"].to_numpy(), ppfd=forcing_data["ppfd"].to_numpy()
)

# Subdaily P Model
sd_scaler = SubdailyScaler(forcing_data["time"].to_numpy().astype("datetime64"))
sd_scaler.set_window(np.timedelta64("12", "h"), half_width=np.timedelta64("15", "m"))

subdaily_pmod = SubdailyPModel(
    env=pmod_env,
    fs_scaler=sd_scaler,
    fapar=forcing_data["fapar"].to_numpy(),
    ppfd=forcing_data["ppfd"].to_numpy(),
)
```

## Convert to Two Leaf model

The code below is a draft implementation of the Two Leaf conversion class.

```{code-cell} ipython3
class TwoLeafAssimilation:
    def __init__(
        self, pmodel: PModel | SubdailyPModel, irrad: TwoLeafIrradiance, LAI: np.array
    ):

        # A load of needless inconsistencies in the object attribute names - to be resolved!
        vcmax_pmod = (
            pmodel.vcmax if isinstance(pmodel, PModel) else pmodel.subdaily_vcmax
        )
        vcmax25_pmod = (
            pmodel.vcmax25 if isinstance(pmodel, PModel) else pmodel.subdaily_vcmax25
        )
        optchi_obj = pmodel.optchi if isinstance(pmodel, PModel) else pmodel.optimal_chi
        core_const = (
            pmodel.core_const if isinstance(pmodel, PModel) else pmodel.env.core_const
        )

        # Generate extinction coefficients to express the vertical gradient in photosynthetic
        # capacity after the equation provided in Figure 10 of Lloyd et al. (2010):

        kv_Lloyd = np.exp(
            0.00963 * vcmax_pmod - 2.43
        )  # KB: Here I opt for vcmax rather than Vcmax25

        # Calculate carboxylation in the two partitions at standard conditions
        self.Vmax25_canopy = LAI * vcmax25_pmod * ((1 - np.exp(-kv_Lloyd)) / kv_Lloyd)
        self.Vmax25_sun = (
            LAI
            * vcmax25_pmod
            * ((1 - np.exp(-kv_Lloyd - irrad.kb * LAI)) / (kv_Lloyd + irrad.kb * LAI))
        )
        self.Vmax25_shade = self.Vmax25_canopy - self.Vmax25_sun

        # Convert carboxylation rates to ambient temperature using an Arrhenius function
        self.Vmax_sun = self.Vmax25_sun * np.exp(
            64800 * (pmodel.env.tc - 25) / (298 * 8.314 * (pmodel.env.tc + 273))
        )
        self.Vmax_shade = self.Vmax25_shade * np.exp(
            64800 * (pmodel.env.tc - 25) / (298 * 8.314 * (pmodel.env.tc + 273))
        )

        # Now the photosynthetic estimates as V_cmax * mc
        self.Av_sun = self.Vmax_sun * optchi_obj.mc
        self.Av_shade = self.Vmax_shade * optchi_obj.mc

        ## Jmax estimates for sun and shade;
        self.Jmax25_sun = 29.1 + 1.64 * self.Vmax25_sun  # Eqn 31, after Wullschleger
        self.Jmax25_shade = 29.1 + 1.64 * self.Vmax25_shade

        # Temperature correction (Mengoli 2021 Eqn 3b); relevant temperatures given in Kelvin
        self.Jmax_sun = self.Jmax25_sun * np.exp(
            (43990 / 8.314) * (1 / 298 - 1 / (pmodel.env.tc + 273))
        )
        self.Jmax_shade = self.Jmax25_shade * np.exp(
            (43990 / 8.314) * (1 / 298 - 1 / (pmodel.env.tc + 273))
        )

        # Calculate J and Aj for each partition
        self.J_sun = self.Jmax_sun * irrad.I_csun * (1 - 0.15) / (irrad.I_csun + 2.2 * self.Jmax_sun)
        self.J_shade = (
            self.Jmax_shade
            * irrad.I_cshade
            * (1 - 0.15)
            / (irrad.I_cshade + 2.2 * self.Jmax_shade)
        )

        self.Aj_sun = (self.J_sun / 4) * optchi_obj.mj
        self.Aj_shade = (self.J_shade / 4) * optchi_obj.mj

        # Calculate the assimilation in each partition as the minimum of Aj and Av,
        # in both cases clipping when the sun is below the angle of obscurity
        Acanopy_sun = np.minimum(self.Aj_sun, self.Av_sun)
        self.Acanopy_sun = np.where(
            irrad.beta_angle < irrad.solar_obscurity_angle, 0, Acanopy_sun
        )
        Acanopy_shade = np.minimum(self.Aj_shade, self.Av_shade)
        self.Acanopy_shade = np.where(
            irrad.beta_angle < irrad.solar_obscurity_angle, 0, Acanopy_shade
        )

        self.gpp = core_const.k_c_molmass * self.Acanopy_sun + self.Acanopy_shade

        # # and we account for canopy respiration
        # gpp_canopy = if_else(sEa > 0.02,
        #                   Acanopy_sun + Acanopy_shade - (rd_pmod * LAI),
        # #                   0))
```

With that class, we can now convert the two big leaf predictions.

```{code-cell} ipython3
# Calculate two leaf, two stream values.

two_leaf_assim_standard = TwoLeafAssimilation(
    irrad=two_leaf_irrad, pmodel=standard_pmod, LAI=forcing_data["LAI"].to_numpy()
)

two_leaf_assim_subdaily = TwoLeafAssimilation(
    irrad=two_leaf_irrad, pmodel=subdaily_pmod, LAI=forcing_data["LAI"].to_numpy()
)
```

## Model comparisons

The plots below then show comparisons between the different implementations.

### Standard PModel time series

This shows a short 3 day time slice of the `pyrealm` Big Leaf and Two Leaf predictions
and the FLUXNet `GPP_DT_CUT_REF` values

```{code-cell} ipython3
plot_slice = slice(
    np.where(forcing_data["time"] == np.datetime64('2014-08-20'))[0][0],
    np.where(forcing_data["time"] == np.datetime64('2014-08-23'))[0][0],
)

plt.plot(
    forcing_data["time"][plot_slice],
    standard_pmod.gpp[plot_slice],
    label="Big Leaf P Model",
)
plt.plot(
    forcing_data["time"][plot_slice],
    two_leaf_assim_standard.gpp[plot_slice],
    label="Two Leaf P Model",
)
plt.plot(
    forcing_data["time"][plot_slice],
    12.014 * forcing_data["gpp_fluxnet"][plot_slice],
    label="FLUXNet",
)
plt.legend(loc="upper center", ncols=3, frameon=False);
```

### Subdaily PModel time series

The same three days, but this time showing the `pyrealm` subdaily implementation for Big
Leaf and Two Leaf. The predictions from the JAMES subdaily model are also shown along
with the FLEXNet values.

```{code-cell} ipython3
plt.plot(
    forcing_data["time"][plot_slice],
    subdaily_pmod.gpp[plot_slice],
    label="Big Leaf P Model",
)
plt.plot(
    forcing_data["time"][plot_slice],
    two_leaf_assim_subdaily.gpp[plot_slice],
    label="Two Leaf P Model",
)
plt.plot(
    forcing_data["time"][plot_slice],
    12.014 * forcing_data["gpp_fluxnet"][plot_slice],
    label="FLUXNet",
)
plt.plot(
    forcing_data["time"][plot_slice],
    12.014 * forcing_data["GPP_JAMES"][plot_slice],
    label="JAMES",
)


plt.legend(loc="upper center", ncols=4, frameon=False);
```

### Comparing Big Leaf and Two Leaf

The plots below compare the complete set of annual predictions from `pyrealm` for the
Big Leaf and Two Leaf models.

```{code-cell} ipython3
fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(8,4), sharex=True, sharey=True)

panels = [
    {"x":standard_pmod.gpp, 
     "y": two_leaf_assim_standard.gpp, 
     "xlab": "GPP Standard PModel BigLeaf", 
     "ylab": "GPP Standard PModel TwoLeaf"
    },
    {"x":subdaily_pmod.gpp, 
     "y": two_leaf_assim_subdaily.gpp, 
     "xlab": "GPP Subdaily PModel BigLeaf", 
     "ylab": "GPP Subdaily PModel TwoLeaf"
    },
]

for panel_info, ax in zip(panels, axes.flatten()):
    
    both_non_zero = np.logical_and(panel_info['x'] > 0 , panel_info['y'] > 0)
    
    ax.hexbin(
        panel_info['x'][both_non_zero], 
        panel_info['y'][both_non_zero], 
        mincnt=1, 
        gridsize=40, 
        xscale='log', 
        yscale='log'
    )
    ax.set_xlabel(panel_info['xlab'])
    ax.set_ylabel(panel_info['ylab'])
    ax.plot([0.1,700], [0.1,700], c='r')

plt.tight_layout()
```

### Comparing with other estimates

The following plots then compare the `pyrealm` predictions from the Standard and
Subdaily models using both the Big Leaf and Two Leaf approaches to the FLUXNet GPP
estimates for the site and the JAMES GPP calculations. The strong fit between the
`pyrealm` Big Leaf Subdaily predictions and the JAMES predictions are expected - they're
effectively the same model, with small extensions to include acclimation of the $\xi$
parameter.

```{code-cell} ipython3
fig, axes = plt.subplots(ncols=2, nrows=4, figsize=(9,16), sharex=True, sharey=True)

panels = [
    {"x": 12.014 * forcing_data['gpp_fluxnet'], 
     "y":standard_pmod.gpp, 
     "xlab": "FLUXNet GPP_DT_CUT_REF",
     "ylab": "GPP Standard PModel Big Leaf", 
    },
    {"x": 12.014 * forcing_data['GPP_JAMES'], 
     "y":standard_pmod.gpp, 
     "xlab": "JAMES Subdaily",
     "ylab": "GPP Standard PModel Big Leaf", 
    },
    {"x": 12.014 * forcing_data['gpp_fluxnet'], 
     "y":two_leaf_assim_standard.gpp, 
     "xlab": "FLUXNet GPP_DT_CUT_REF",
     "ylab": "GPP Standard PModel Two Leaf", 
    },
    {"x": 12.014 * forcing_data['GPP_JAMES'], 
     "y":two_leaf_assim_standard.gpp, 
     "xlab": "JAMES Subdaily",
     "ylab": "GPP Standard PModel Two Leaf", 
    },
    {"x": 12.014 * forcing_data['gpp_fluxnet'], 
     "y":subdaily_pmod.gpp, 
     "xlab": "FLUXNet GPP_DT_CUT_REF",
     "ylab": "GPP Subdaily PModel Big Leaf", 
    },
    {"x": 12.014 * forcing_data['GPP_JAMES'], 
     "y":subdaily_pmod.gpp, 
     "xlab": "JAMES Subdaily",
     "ylab": "GPP Subdaily PModel Big Leaf", 
    },
    {"x": 12.014 * forcing_data['gpp_fluxnet'], 
     "y":two_leaf_assim_subdaily.gpp, 
     "xlab": "FLUXNet GPP_DT_CUT_REF",
     "ylab": "GPP Subdaily PModel Two Leaf", 
    },
    {"x": 12.014 * forcing_data['GPP_JAMES'], 
     "y":two_leaf_assim_subdaily.gpp, 
     "xlab": "JAMES Subdaily",
     "ylab": "GPP Subdaily PModel Two Leaf", 
    },
]

for panel_info, ax in zip(panels, axes.flatten()):
    
    both_non_zero = np.logical_and(panel_info['x'] > 0 , panel_info['y'] > 0)
    
    ax.hexbin(
        panel_info['x'][both_non_zero], 
        panel_info['y'][both_non_zero], 
        mincnt=1, 
        gridsize=40, 
        xscale='log', 
        yscale='log'
    )
    ax.set_xlabel(panel_info['xlab'])
    ax.set_ylabel(panel_info['ylab'])
    ax.plot([0.1,700], [0.1,700], c='r')

plt.tight_layout()
```

### Vcmax comparisons from Sun and Shade

```{code-cell} ipython3
plt.plot(
    forcing_data["time"][plot_slice],
    subdaily_pmod.subdaily_vcmax[plot_slice],
    label="Subdaily Vcmax",
)
plt.plot(
    forcing_data["time"][plot_slice],
    two_leaf_assim_subdaily.Vmax_sun[plot_slice],
    label="Sun Vcmax",
)
plt.plot(
    forcing_data["time"][plot_slice],
    two_leaf_assim_subdaily.Vmax_shade[plot_slice],
    label="Shade Vcmax",
)

plt.legend(loc="upper center", ncols=3, frameon=False);
```

### Export calculations

The code below exports key calculations so that they can be compared to Keith's R
implementation of the calculations.

```{code-cell} ipython3
python_outputs = pd.DataFrame({
    'time': forcing_data['time'],
    'gpp_pmod': (
        subdaily_pmod.gpp / subdaily_pmod.env.core_const.k_c_molmass # as µmol not µg
    ),
    "gammastar_pmod": subdaily_pmod.env.gammastar,
    "kmm_pmod": subdaily_pmod.env.kmm,
    "ci_pmod": subdaily_pmod.optimal_chi.ci,
    "vcmax_pmod": subdaily_pmod.subdaily_vcmax,
    "jmax_pmod": subdaily_pmod.subdaily_jmax,
    "vcmax25_pmod": subdaily_pmod.subdaily_vcmax25,
    # "rd_pmod": subdaily_pmod.rd,
    # "kv_Lloyd": ,
    "Icsun": two_leaf_irrad.I_csun,
    "Icshade": two_leaf_irrad.I_cshade,
    # "sEa": ,
    "kb": two_leaf_irrad.kb,
    "Vmax25_canopy": two_leaf_assim_subdaily.Vmax25_canopy,
    "Vmax25_sun":  two_leaf_assim_subdaily.Vmax25_sun,
    "Vmax25_shade":  two_leaf_assim_subdaily.Vmax25_shade,
    "Vmax_sun":  two_leaf_assim_subdaily.Vmax_sun,
    "Vmax_shade": two_leaf_assim_subdaily.Vmax_shade,
    "Av_sun": two_leaf_assim_subdaily.Av_sun,
    "Av_shade": two_leaf_assim_subdaily.Av_shade,
    "Jmax25_sun": two_leaf_assim_subdaily.Jmax25_sun,
    "Jmax25_shade": two_leaf_assim_subdaily.Jmax25_shade,
    "Jmax_sun": two_leaf_assim_subdaily.Jmax_sun,
    "Jmax_shade": two_leaf_assim_subdaily.Jmax_shade,
    "J_sun": two_leaf_assim_subdaily.J_sun,
    "J_shade": two_leaf_assim_subdaily.J_shade,
    "Aj_sun": two_leaf_assim_subdaily.Aj_sun,
    "Aj_shade": two_leaf_assim_subdaily.Aj_shade,
    "Acanopy_sun": two_leaf_assim_subdaily.Acanopy_sun,
    "Acanopy_shade": two_leaf_assim_subdaily.Acanopy_shade,
    "gpp_canopy": (
        two_leaf_assim_subdaily.gpp / subdaily_pmod.env.core_const.k_c_molmass
    ), # as µmol not µg
})

python_outputs.to_csv('two_leaf_python_implementation_outputs.csv', index=False)
```
