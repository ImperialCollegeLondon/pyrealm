---
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
---

# The subdaily P Model

The standard [P Model](../pmodel_details/pmodel_overview.md) assumes that plants are at
equilibrium with their environmental conditions. This is a reasonable approximation when
using forcing variables that capture average conditions over longer time scales such as
weekly, monthly or coarser time steps. However, at finer temporal scales - and
particularly when trying to describe photosynthetic behaviour at subdaily timescales -
this assumption breaks down. Plants cannot instantaneously respond optimally to changing
conditions and it is essential to account for fast and slow responses to changing
environmental conditions.

This page gives an overview of the calculations for the subdaily form of the P Model
{cite:p}`mengoli:2022a` along with links to further details of the core
components of the model. The `pyrealm` implementation largely follows the weighted
average method of {cite:t}`mengoli:2022a`, but is modified to include slow responses in
$\xi$. It has also been extended to allow selection of different methods for calculating
optimal $\chi$, estimating quantum yield ($\phi_0$), limitation factors and Arrhenius
scaling.

It may be useful to read this alongside:

* The [worked examples](worked_example.md) of using `pyrealm` to fitting the subdaily P
  Model.
* Advice on using the subdaily P Model with [missing
  data](subdaily_model_and_missing_data.md): because the subdaily P Model includes
  iterated calculation across a time-series, the model needs to be adapted to handle
  missing values.
* The [subdaily P Model calculations](./subdaily_calculations.md) page, which walks
  through each step of the calculation workflow shown below using `pyrealm` code.
* The descriptions of the key Python classes used to fit the standard P Model:
  {class}`~pyrealm.pmodel.pmodel_environment.PModelEnvironment`,
  {class}`~pyrealm.pmodel.acclimation.AcclimationModel`, and
  {class}`~pyrealm.pmodel.pmodel.SubdailyPModel`.

The subdaily model differentiates between responses to changing conditions at two
speeds:

Fast responses
: At timescales of minutes,  include the opening and closing of stomata in response to
  changing conditions and changes in enzyme kinetics in response to changing conditions.

Slow responses
: At timescales of around a fortnight, include the adaptation of three key parameters
  that acclimate over time to changing environmental conditions.

    * $\xi$: the sensitivity of the optimal $\chi$ ratio to the vapour pressure deficit.
    * $V_{cmax25}$: the maximum rate of carboxylation at standard temperature,
    * $J_{max25}$: the maximum rate of electron transport at standard temperature, and

  The $\xi$ parameter captures the plant response to changing hygroscopic conditions.
  The other parameters control the shape of the $A$/$C_i$ curve for photosynthesis:
  $V_{cmax25}$ constrains the Rubisco-limited assimilation rate ($A_c$) and $J_{max25}$
  constrains the electron transport rate limited assimilation rate ($A_J$).

The diagram below shows the workflow used for the calculation of the subdaily P Model:
the input forcing data is show in blue, internal variables are shown in green and the
core shared calculations are shown in red.

<!-- markdownlint-disable MD033 -->
<!--
The iframe below is generated from the File > Embed menu in drawio. It has the
advantage of providing a zoomable cleaner interface for the diagram that supports
tooltips
-->

<iframe frameborder="0" style="width:100%;height:800px;" src="https://viewer.diagrams.net/?lightbox=0&highlight=0000ff&nav=1&title=pmodel.drawio&dark=auto#Uhttps%3A%2F%2Fraw.githubusercontent.com%2FImperialCollegeLondon%2Fpyrealm%2Fdevelop%2Fdocs%2Fsource%2Fusers%2Fpmodel%2Fsubdaily_details%2Fsubdaily_pmodel.drawio"></iframe>

The implementation has the following steps:

1. The [subdaily photosynthetic
  environment](../shared_components/photosynthetic_environment) for the data is calculated
  as for the standard P Model. This estimates the fast responses of $\Gamma^*$,
  $K_{mm}$, $\eta^*$, and $c_a$.

2. An [acclimation model](acclimation.md#the-acclimation-model) is then defined that
  sets a window of observations during the day that define the conditions that the slow
  responses will acclimate towards. Thease are typically noon conditions that optimise
  light use efficiency during the daily period of highest photosynthetic photon flux
  density (PPFD). This model is used to calculate a daily time series of average
  conditions during this acclimation window.

3. A [standard P model](../pmodel_details/pmodel_overview.md) is used to estimate
   optimal behaviour during the daily acclimation conditions.

4. The **daily optimal values** of $\xi$, $V_{cmax}$ and $J_{max}$ are then extracted
  from this model. An [Arrhenius scaling](../shared_components/arrhenius.md) is applied
  to give standardised values of $V_{cmax25}$ and $J_{max25}$.

5. The daily optimal values are then passed through an acclimation function, which uses
   an exponential weighted mean to estimate the **daily realised values** of these
   parameters, given the speed with which they react to change.

6. The daily realised values are then [resampled to the subdaily
   timescale](acclimation.md#interpolation-of-realised-values-to-subdaily-timescales) to
   give subdaily realised values. The subdaily realised values of  $V_{cmax25}$ and
   $J_{max25}$ are rescaled to the actual subdaily temperatures using Arrhenius scaling
   to give realised value for $V_{cmax}$ and $J_{max}$.

7. The [quantum yield of photosynthesis](../shared_components/quantum_yield.md)
   ($\phi_0$) and [optimal $\chi$](../shared_components/optimal_chi.md) are then
   calculated for subdaily conditions, but $\chi$ is constrained by using the slowly
   responding realised values of $\xi$.

8. Unlike the standard model, which can calculate light use efficiency more directly,
   the subdaily model needs to use realised $V_{cmax}$ and $J_{max}$ to calculate
   the maximum rate of assimilation given limitation of the electron transfer ($A_j$)
   and carboxylation ($A_c$) pathways. GPP is then calculated as the minimum of those
   two rates.
