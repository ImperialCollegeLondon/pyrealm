---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3 (ipykernel)
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

<!-- markdownlint-disable-next-line MD041 -->
(pmodel_overview)=

# The Standard P Model

This page gives an overview of the calculations for the standard form of the P Model
{cite:p}`Prentice:2014bc,Wang:2017go` along with links to further details of the core
components of the model. It may be useful to read this alongside:

* The [worked examples](worked_examples) of using `pyrealm` to fitting the Standard P
  Model.
* The overview of the [expected predictions](./envt_variation_outputs.md) of the model,
  given forcing data within normal bounds.
* The descriptions of the key Python classes used to fit the standard P Model:
  {class}`~pyrealm.pmodel.pmodel_environment.PModelEnvironment` and
  {class}`~pyrealm.pmodel.pmodel.PModel`.

:::{important}

The standard form of the P Model is intended to work with data on greater than weekly
timescales, where the plants are assumed to be roughly at equilibrium with their
environment. At faster time scales, the [subdaily form of the P
Model](../subdaily_details/subdaily_overview.md) accounts for more slowly responding
systems that introduce acclimation lags into the model calculations.

:::

The diagram below shows the workflow used for the calculation of the standard P Model:
the input forcing data is show in blue, internal variables are shown in green and the
core shared calculations are shown in red.

<!-- markdownlint-disable MD033 -->
<!--
The iframe below is generated from the File > Embed menu in drawio. It has the
advantage of providing a zoomable cleaner interface for the diagram that supports
tooltips
-->

<iframe frameborder="0" style="width:100%;height:650px;" src="https://viewer.diagrams.net/?lightbox=0&highlight=0000ff&nav=1&title=pmodel.drawio&dark=auto#Uhttps%3A%2F%2Fraw.githubusercontent.com%2FImperialCollegeLondon%2Fpyrealm%2F438-revise-the-p-model-documentation%2Fdocs%2Fsource%2Fusers%2Fpmodel%2Fpmodel_details%2Fpmodel.drawio"></iframe>

<!-- markdownlint-enable MD033 -->

In overview:

1. The forcing variables are used to calculate the [photosynthetic
  environment](../shared_components/photosynthetic_environment.md) for the model,
  including the:

    * photorespiratory compensation point ($\Gamma^*$),
    * Michaelis-Menten coefficient for photosynthesis ($K_{mm}$),
    * relative viscosity of water ($\eta^*$),
    * the partial pressure of $\ce{CO2}$ in ambient air ($c_a$), and
    * the absorbed irradiance ($I_{abs}$).

2. The photosynthetic environment is then used to calculate [optimal
   chi](../shared_components/optimal_chi.md), given the method set by the
   `method_optchi` argument, to calculate:

    * the ratio of internal to ambient $\ce{CO2}$ partial pressure ($\chi$),
    * the internal $\ce{CO2}$ partial pressure ($c_i$),
    * the $\ce{CO2}$ limitation factors to both light assimilation ($m_j$) and
      carboxylation ($m_c$) along with their ratio ($m_j / m_c$).

    The term $m_j$ is at the heart of the P model and describes the trade off between
    carbon dioxide capture and water loss in photosynthesis. Given the environmental
    conditions, a leaf will adjust its stomata to a value of ($\chi$) that optimises
    this trade off. When $\chi$ is less than one, the partial pressure inside of
    $\ce{CO2}$ inside the leaf is lowered and $m_j$ captures the resulting loss in light
    use efficiency.

3. The photosynthetic environment is also used to calculate the [quantum
   yield](../shared_components/quantum_yield.md) of photosynthesis ($\phi_0$), given the
   method set by the `method_kphio` argument.

4. Theory suggests that $m_j$ and $m_c$ should be further limited ($J_max$ limitation),
   with different approaches proposed {cite}`Wang:2017go,Smith2019dv`. The calculation
   of [limitation factors](../shared_components/jmax_limitation.md), given the
   method set by the `method_jmaxlim` argument, represents these alternative corrections
   as:

    * a limitation term on the electron transfer rate ($f_j$), and
    * a similar limitation term on the carboxylation capacity ($f_v$).

5. From these values, and the molar mass of carbon ($M_C$), the light use efficiency of
   photosynthesis can then be calculated as:

    $$
      \text{LUE} = M_C \cdot \phi_0 \cdot  m_j \cdot f_v
    $$

    The gross primary productivity (GPP) is then simply the product of LUE and the
    absorbed irradiation ($I_abs = f_{APAR} \cdot PPFD$).

6. The approach above is equivalent to directly calculating the minimum of the
   assimilation rates given limitation of the electron transfer pathway ($A_j$) and
   limitation of the carboxylation pathway ($A_c$). However the model does still
   estimate the maximum rates of electron transfer ($J_{max}$) and carboxylation
   capacity ($V_{cmax}$) and then uses [Arrhenius
   scaling](../shared_components/arrhenius.md) to estimate those values at standard
   temperature  ($J_{max25}$) and carboxylation
   capacity ($V_{cmax25}$).
