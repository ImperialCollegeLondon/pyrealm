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

# The demography module

The functionality of the demography module is split into the following submodules

* The [`flora` module](./flora.md) that defines the set of plant functional traits used
  in demographic modelling and the classes used to represent those traits.
* The [`t_model` module](./t_model.md) module that implements the allometric and
  allocation equations of the T Model {cite}`Li:2014bc`.
* The [`crown` module](./crown.md) that implements a three dimensional model of crown
  shape taken from the Plant-FATE model {cite}`joshi:2022a`.
* The [`community` module](./community.md) that implements a plant community model using
  size-structured cohorts.
* The [`canopy` module](./canopy.md) that generates a model of the canopy structure for
  a community, based on the Perfect Plasticity Approximation model {cite}`purves:2008a`.
