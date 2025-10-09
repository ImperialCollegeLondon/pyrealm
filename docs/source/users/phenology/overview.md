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
---

# The phenology module

The {mod}`~pyrealm.phenology` module implements tools for predicting phenological time
series of {term}`FAPAR` and {term}`LAI`. There are two main stages in this process:

1. Annual maximum values of FAPAR and LAI for a location can be predicted by comparing
   constraints on potential gross primary productivity arising from either water
   limitation or energy limitation {cite:p}`cai:2025a`. This is implemented in `pyrealm`
   as the {class}`~pyrealm.phenology.fapar_limitation.FaparLimitation` class and is
   described in the [introduction to calculating maximum FAPAR](./fapar_limitation.py).

2. The maximum annual FAPAR and LAI can then be combined with daily predictions of
   potential GPP to predict time series of realised FAPAR and LAI through the year,
   [TBD].
