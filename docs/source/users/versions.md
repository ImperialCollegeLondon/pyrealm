---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
language_info:
  codemirror_mode:
    name: ipython
    version: 3
  file_extension: .py
  mimetype: text/x-python
  name: python
  nbconvert_exporter: python
  pygments_lexer: ipython3
  version: 3.11.9
---

# Versions and migration

The `pyrealm` package is still being actively developed and the functionality in the
package is being extended and revised. This document is describes the main changes in
functionality between major versions and provides code migration notes for changes in
the API of `pyrealm` code.

## Migration from 1.0.0 to version 2.0.0

The versions of `PModel` and `SubdailyPModel` in `pyrealm` version 1.0.0 were based on
the original implementation of the `rpmodel` package {cite}`Stocker:2020dh` and the R
code supporting {cite}`mengoli:2022a`, respectively. The two implementations:

* were inconsistent in the names used for attributes common across the two models,
* contained duplicated internal code, and
* differed in the default settings.

In addition, several core methods used within the calculations of both models had been
rewritten to provide a more flexible and extensible framework for new research methods.
So, version 2.0.0 provides a complete reworking of the package, with a particular focus
on better integrating the  {class}`~pyrealm.pmodel.pmodel.PModel` and
{class}`~pyrealm.pmodel.pmodel.SubdailyPModel` classes. This has led to a large
number of breaking changes in the API. As the package uses [semantic
versioning](https://semver.org/), these changes to the API require that new releases be
made under a new major version.

```{warning}

We will be publishing a series of "release candidates" of the 2.0.0 package. These will
be used to identify issues with the current API and try to stabilise a new API. The
content of version 2.0.0 is not yet finalised, so these release candidates may also add
new functionality.

```

The main user facing changes are shown below, but do also look at the [log of
changes](#changes-log) for more detail.

### The PModelEnvironment class

The `PModel` and `SubdailyPModel` were inconsistent in the API for providing FAPAR and
PPFD. Both of these variables are now required parts of the
{class}`~pyrealm.pmodel.pmodel_environment.PModelEnvironment`, with a default value of
one. As a result:

* The `PModel` no longer requires use of the `PModel.estimate_productivity()` method to
  estimate GPP and this has been redacted.
* The `fapar` and `ppfd` arguments have been removed from `SubdailyPModel()`.

Internally, the `PModelEnvironment` now automatically calculates temperature in Kelvin
and can be used more flexibly with additional forcing variables for new methods.

### The quantum yield of photosynthesis ($\phi_0$)

The {class}`~pyrealm.pmodel.pmodel.PModel` in 1.0.0 supported fixed and temperature
dependent $\phi_0$ through the `do_ftemp_kphio` argument. The
{class}`~pyrealm.pmodel.pmodel.SubdailyPModel` only supported temperature dependent
$\phi_0$. Both classes also used the `kphio` argument to override the default values
and these defaults _differed_ between the two models: `PModel` used default values taken
from {cite:t}`Stocker:2020dh` that depended on `do_ftemp_kphio` and `SubdailyPModel`
used the theoretical maximum value of 1/8.

* Both classes now use `method_kphio` and `reference_kphio` and the arguments `kphio`
  and `do_ftemp_kphio` have been removed.
* The `method_kphio` argument now uses an extendable set of options for calculating
  $\phi_0$ so that new methods can be added seamlessly. Currently, `pyrealm` provides
  'fixed' and 'temperature' along with an experimental 'sandoval' method that models the
  effects of growth temperature and aridity on $\phi_0$.
* All of the available method choices to `PModel` and `SubdailyPModel` now use $\phi_0 =
  1/8$ as the default value but this can be overriden using `reference_kphio`.

Since this involves a change in the default behaviour that leads to different
predictions, the `PModel` class in version 2.0.0 issues a warning to alert users.

### Arrhenius scaling of $J_{max}$ and $V_{cmax}$

In 1.0.0, {class}`~pyrealm.pmodel.pmodel.PModel` used an Arrhenius relationship with a
peak at intermediate temperatures {cite}`Kattge:2007db` to calculate $J_{max25}$ and
$V_{cmax25}$, although in practice the implementation did not exhibit the correct peaked
form. In contrast, the {class}`~pyrealm.pmodel.pmodel.SubdailyPModel` used a simple
Arrhenius scaling without a peak.

* Both classes now take the `method_arrhenius` argument to specify the form of this
  scaling.
* Both classes now default to using the `simple` method of Arrhenius scaling. We
  **strongly** recommend the use of this method over the experimental `kattge_knorr`
  method.

### Method choices in the Subdaily PModel

As noted above, in version 1.0.0, the `SubdailyPModel` was fixed to use the standard C3
calculation of optimal $\chi$, temperature-dependent estimation of $\phi_0$, simple
Arrhenius scaling and estimation of $J_{max}$ and $V_{cmax}$ limitation following
{cite:t}`Wang:2017go`. The class now accepts the four method arguments used by the
standard `PModel` (`method_optchi`, `method_kphio`, `method_arrhenius` and
`method_jmaxlim`).

### The Acclimation Model

In 1.0.0, the {class}`~pyrealm.pmodel.pmodel.SubdailyPModel` model required a large
number of options that were used to set the details of the acclimation model to be used.

* The `fs_scaler` argument was used to provide a `SubdailyScaler` object that
  established the timing of the observations and provided scaling between the daily and
  subdaily scales.
* The `alpha`, `allow_holdover`, `allow_partial_data` and `fill_kind` arguments were
  then used to modify the functions used to generate the acclimation model.

All of this functionality has now been brought together into a single
{class}`~pyrealm.pmodel.acclimation.AcclimationModel` class that integrates all of those
settings into a single object. In addition:

* The standalone `memory_effect` function has become the
  {class}`AcclimationModel.apply_acclimation<pyrealm.pmodel.acclimation.AcclimationModel.apply_acclimation>`
  method, although the underlying exponential moving average function is now available
  as the {meth}`~pyrealm.core.utilities.exponential_moving_average` function.
* The standalone `convert_pmodel_to_subdaily` function is now the
  {meth}`PModel.to_subdaily<pyrealm.pmodel.pmodel.PModel.to_subdaily>` method.

### Code comparison

The tabs below show the calculation of a `PModel` and `SubdailyPModel` using version
1.0.0 and version 2.0.0 of `pyrealm`.

`````{tab-set}
````{tab-item} pyrealm 1.0.0
```{code-block} ipython3
# Create the PModelEnvironment
pm_env = PModelEnvironment(
  tc=tc,
  patm=patm,
  vpd=vpd,
  co2=co2
)

# Fit the standard P Model
standard_model = PModel(
  env=pm_env,
  method_optchi="prentice14",
  do_ftemp_kphio=False,
  kphio=1 / 8
)
pmodC3.estimate_productivity(fapar=fapar, ppfd=ppfd)

# Create the SubdailyScaler
fsscaler = SubdailyScaler(datetimes)
fsscaler.set_window(
    window_center=np.timedelta64(12, "h"),
    half_width=np.timedelta64(1, "h"),
)

# Fit the subdaily model
subdailyC3 = SubdailyPModel(
    env=pm_env,
    kphio=1 / 8,
    method_optchi="prentice14",
    fapar=fapar,
    ppfd=ppfd,
    fs_scaler=fsscaler,
    alpha=1 / 15,
    allow_holdover=True,
)
```
````
````{tab-item} pyrealm 2.0.0
```{code-block} ipython
# Create the PModelEnvironment, including FAPAR and PPFD
pm_env = PModelEnvironment(
  tc=tc,
  patm=patm,
  vpd=vpd,
  co2=co2,
  fapar=fapar,
  ppfd=ppfd,
)

# Fit the standard P Model - 'estimate_productivity' not required
standard_model = PModel(
  env=pm_env,
  method_optchi="prentice14",
  method_kphio="fixed",
  reference_kphio=1 / 8,  # Although this is now the default.

)

# Create the acclimation model - merging acclimation functions into a common class
acclim_model = AcclimationModel(
  datetimes,
  alpha=1 / 15,
  allow_holdover=True
)
acclim_model.set_window(
    window_center=np.timedelta64(12, "h"),
    half_width=np.timedelta64(1, "h"),
)

# Fit the subdaily model - which now accepts all of the alternative method
# arguments  used by the PModel class.
subdaily_model = SubdailyPModel(
    env=pm_env,
    acclim_model=acclim_model,
    method_kphio="fixed",
    method_optchi="prentice14",
    reference_khio=1 / 8,  # Again, this is the default.
)
```
````
`````

### Duplication of results from version 1.0

The default settings in version 2.0 have been chosen to give appropriate and consistent
predictions. If you do need to duplicate the exact values calculated under version 1.0,
then follow the code examples below. We **do not recommend** using these settings but
they may be useful in validating code migration. Note that setting
``method_arrhenius="kattge_knorr"` is only required to duplicate predictions of
$V_{cmax25}$ and $J_{max25}$.

```{code-block} ipython3
# Create the PModelEnvironment
pm_env = PModelEnvironment(
    tc=tc,
    patm=patm,
    vpd=vpd,
    co2=co2
    fapar=fapar,
    ppfd=ppfd,
    mean_growth_temperature=tc,
)

# With temperature dependent phi_0
ret = PModel(
    env = pm_env,
    method_kphio="temperature",
    method_arrhenius="kattge_knorr",
    reference_kphio=0.081785,
)

# Without temperature dependent phi_0
ret = PModel(
    env = pm_env,
    method_kphio="fixed",
    method_arrhenius="kattge_knorr",
    reference_kphio=0.049977,
)
```

### Supporting functions for the P Model

The {mod}`pyrealm.pmodel.functions` module provides a set of functions specific to the
calculations of the P Model and SubdailyPModel. Many of the functions used
`pmodel_const` and `core_const` arguments to pass in constant values, but all are now
unpacked so that the specific constants needed for each function are clear in the
function signatures. In addition, the `calc_ftemp_kphio` and `calc_ftemp_inst_vcmax`
functions have been removed as they provided specific configurations of the more general
`calculate_simple_arrhenius_factor` and `calculate_kattge_knorr_arrhenius_factor` that
now replace them in the module.

## Changes Log

````{dropdown} Changes summary for pyrealm
```{include} ../../../CHANGES.md
    :start-line: 2
```
````
