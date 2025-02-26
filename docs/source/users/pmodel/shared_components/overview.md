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

# Shared components in the P Model module

The standard and subdaily forms of the P Model share a great deal of theory and many
`pyrealm` code structures. This section of the documentation describes the key shared
concepts and tools used for fitting both forms of the P Model. These components are:

* Calculation of the [photosynthetic
  environment](./photosynthetic_environment), which calculates critical
  photosynthetic variables from the model forcing variable.
* Calculation of [optimal chi](./optimal_chi), defining the optimal ratio
  of carbon dioxide partial pressure inside the leaf compared to the surrounding air.
  This value captures the trade off between water loss and carbon acquisition.
* Estimation of [quantum yield efficiency](./quantum_yield), which sets
  the efficiency with which a plant converts absorbed light radiation into captured
  carbon.
* Estimation of [rate limitation](./jmax_limitation) on the maximum
  values of the electron transfer rate and carboxylation capacity, sometimes called
  $J_{max}$ limitation.
* The form of the [Arrhenius scaling](./arrhenius) used to
  calculate the electron transfer rate and carboxylation capacity at standard
  temperatures. This is central to the Subdaily P Model as the variables $J_{max}$ and
  $V_{cmax}$ need converted to a standard temperature to map predictions from the daily
  scale back to the subdaily scale.
* Approaches to the impacts of [soil moisture stress](./soil_moisture).
* The behaviour of core P Model equations with [extreme forcing
  values](./extreme_values).
