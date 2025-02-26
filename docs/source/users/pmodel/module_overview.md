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

# The P Model module

The {mod}`~pyrealm.pmodel` module implements the P Model and extensions. The P Model
implements an eco-evolutionary optimality (EEO) theory  of plant productivity: that
plants optimise their behaviour to balance the costs and gains of photosynthesis. The
module provides implementations of two forms of the P Model:

* The ['standard' P Model](pmodel_details/pmodel_overview)
  {cite:p}`Prentice:2014bc,Wang:2017go,Stocker:2020dh`.

* The [subdaily P Model](subdaily_details/subdaily_overview), extends the P Model to
  incorporates acclimation of photosynthetic pathways to changing environmental
  conditions {cite}`mengoli:2022a`. This improves the performance of the P Model at fine
  temporal resolutions, by accounting for lags in the adoption of optimal behaviour.

These two forms of the P Model share a great deal of theory and many `pyrealm` code
structures. Key shared structures for fitting either P Model are:

* Calculation of the [photosynthetic
  environment](shared_components/photosynthetic_environment), which calculates critical
  photosynthetic variables from the model forcing variable.
* Calculation of [optimal chi](shared_components/optimal_chi), defining the optimal ratio
  of carbon dioxide partial pressure inside the leaf compared to the surrounding air.
  This value captures the trade off between water loss and carbon acquisition.
* Estimation of [quantum yield efficiency](shared_components/quantum_yield), which sets
  the efficiency with which a plant converts absorbed light radiation into captured
  carbon.
* Estimation of [rate limitation](shared_components/jmax_limitation) on the maximum
  values of the electron transfer rate and carboxylation capacity, sometimes called
  $J_{max}$ limitation.
* The form of the [Arrhenius scaling](./shared_components/arrhenius.md) used to
  calculate the electron transfer rate and carboxylation capacity at standard
  temperatures. This is central to the Subdaily P Model as the variables $J_{max}$ and
  $V_{cmax}$ need converted to a standard temperature to map predictions from the daily
  scale back to the subdaily scale.

  and
* the estimation of [gross primary
  productivity](pmodel_details/envt_variation_outputs.md#productivity-outputs).

* Approaches to the impacts of [soil moisture stress](pmodel_details/soil_moisture).
* The behaviour of P Model equations with [extreme forcing
  values](pmodel_details/extreme_values.md).

In addition to the two P Model forms, this module also provides two extensions that
build on the P Model:

* The process of photosynthesis discriminates between the carbon isotopes occuring in
  air, leaving characteristic isotopic signatures. The [Isotopic
  discrimination](isotopic_discrimination) module estimates the isotopic signatures from
  different P Models.

* The relative advantages of the C3 and C4 photosynthetic pathways differ with
  environmental conditions. The [C3 / C4 plant competition](c3c4model) model uses the
  relative advantage to estimates the expected fraction of C4 plants in a community.
