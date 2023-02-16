---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# The T Model

This module provides a Python implementation of the T-Model (:{cite}`Li:2014bc`), which
provides a physiological model of tree growth given a set of traits on tree growth
scalings and allocation of primary production.

The module uses three key components of the T Model:

1. the {class}`~pyrealm.constants.tmodel_const.TModelTraits` class,
2. the {class}`~pyrealm.tmodel.TTree` class, and
3. the {func}`~pyrealm.tmodel.grow_ttree` function.

## The {class}`~pyrealm.constants.tmodel_const.TModelTraits` class

The T Model depends on a set of 14 traits that are used to describe the geometry of a
tree and the allocation of carbon within the tree and which are listed in the linked
class description.

The class can be used to create a default T Model trait set:

```{code-cell} ipython3
import numpy as np
from pyrealm import tmodel
# A tree using the default parameterisation
traits1 = tmodel.TModelTraits()
print(traits1)
```

It can also be edited to generate different growth patterns:

```{code-cell} ipython3
# A slower growing tree with a higher maximum height
traits2 = tmodel.TModelTraits(a_hd=50, h_max=40)
print(traits2)
```

## The {class}`~pyrealm.tmodel.TTree` class

This class implements the mathematical description of tree growth under the T Model.

There are three stages to generating predictions:

1. initialising a model,
2. setting the stem diameters to calculate tree geometry and mass, and
3. calculating growth predictions for a given estimate of gross primary productivity (GPP).

Note that the {class}`~pyrealm.tmodel.TTree` is not used to model tree growth through
time (see below and {func}`~pyrealm.tmodel.grow_ttree`). It simply calculates the
predictions of the T Model for trees with a given diameter and GPP. Because  it accepts
**arrays** of data, it can be used to very quickly visualise the behaviour of the TModel
for a given set of traits and diameters.

### Initialising a {class}`~pyrealm.tmodel.TTree` object

A {class}`~pyrealm.tmodel.TTree` object is created by providing an array of initial stem
diameters and an optional set of traits as a
{class}`~pyrealm.constants.tmodel_const.TModelTraits` object. If no traits are provided,
the default {class}`~pyrealm.constants.tmodel_const.TModelTraits` settings are used.

```{code-cell} ipython3
# Use a sequence of diameters from sapling to large tree
diameters = np.linspace(0.02, 2, 100)
tree1 = tmodel.TTree(diameters=diameters)  # Using default traits 
tree2 = tmodel.TTree(diameters=diameters, traits=traits2)
```

These inputs are then immediately used to calculate the following properties of the
{class}`~pyrealm.tmodel.TTree` that scale simply with tree diameter:

* Stem diameter (`diameter`)
* Stem height (`height`)
* Crown fraction (`crown_fraction`)
* Crown area (`crown_area`)
* Mass of stem (`mass_stm`)
* Mass of foliage and fine roots (`mass_fol`)
* Mass of sapwood (`mass_swd`)

Using an array of diameter values provides an immediate way to visualise the geometric
scaling resulting from a particular set of plant traits:

```{code-cell} ipython3
:tags: [hide-input]

from matplotlib import pyplot
fig, (ax1, ax2, ax3) = pyplot.subplots(1,3, figsize=(12, 4))
ax1.plot(tree1.diameter, tree1.height, label='traits1')
ax1.plot(tree2.diameter, tree2.height, label='traits2')
ax1.set_title('Height')
ax1.set_xlabel('Stem diameter (m)')
ax1.set_ylabel('Height (m)')
ax1.legend()
ax2.plot(tree1.diameter, tree1.crown_area, label='traits1')
ax2.plot(tree2.diameter, tree2.crown_area, label='traits2')
ax2.set_title('Crown Area')
ax2.set_xlabel('Stem diameter (m)')
ax2.set_ylabel('Crown area (m2)')
ax2.legend()
ax3.plot(tree1.diameter, tree1.mass_stm, label='Stem')
ax3.plot(tree1.diameter, tree1.mass_swd, label='Sapwood')
ax3.plot(tree1.diameter, tree1.mass_fol, label='Leaf and fine root')
ax3.set_yscale('log')
ax3.set_title('Mass (traits1)')
ax3.set_xlabel('Stem diameter (m)')
ax3.set_ylabel('Log Mass (kg)')
ax3.legend()
pyplot.show()
```

### Calculating growth

The {meth}`~pyrealm.tmodel.TTree.calculate_growth` method is used to provide estimates
of gross primary productivity (GPP) to a {class}`~pyrealm.tmodel.TTree` instance in
order to apply the T Model predictions of GPP allocation to plant respiration and
growth.

Running the method populates properties of the {class}`~pyrealm.tmodel.TTree` that
provide estimates of the following growth parameters:

* Gross primary productivity (`gpp_actual`)
* Net primary productivity (`npp`)
* Sapwood respiration (`resp_swd`)
* Fine root respiration (`resp_frt`)
* Foliage maintenance respiration (`resp_fol`)
* Foliage and fine root turnover (`turnover`)
* Diameter increment (`delta_d`)
* Stem mass increment (`delta_mass_stm`)
* Fine root mass increment (`delta_mass_frt`)

The code below calculates growth estimates at each diameter under a constant GPP of 7
TODO - UNITS!.

```{code-cell} ipython3
tree1.calculate_growth(np.array([7]))
tree2.calculate_growth(np.array([7]))
```

```{code-cell} ipython3
:tags: [hide-input]

fig, (ax1, ax2, ax3) = pyplot.subplots(1,3, figsize=(12, 4))
ax1.plot(tree1.diameter, tree1.npp, label='traits1')
ax1.plot(tree2.diameter, tree2.npp, label='traits2')
ax1.set_title('NPP')
ax1.set_xlabel('Net primary productivity (m)')
ax1.set_ylabel('Height (m)')
ax1.legend()
ax2.plot(tree1.diameter, tree1.resp_swd, label='Sapwood')
ax2.plot(tree1.diameter, tree1.resp_frt, label='Fine roots')
ax2.plot(tree1.diameter, tree1.resp_fol, label='Foliage')
ax2.set_title('Respiration (traits1)')
ax2.set_xlabel('Stem diameter (m)')
ax2.set_ylabel('Respiration')
ax2.legend()
ax3.plot(tree1.diameter, tree1.delta_d, label='traits1')
ax3.plot(tree2.diameter, tree2.delta_d, label='traits2')
ax3.set_title('Delta diameter ')
ax3.set_xlabel('Stem diameter (m)')
ax3.set_ylabel('Delta diameter (m)')
ax3.legend()
pyplot.show()
```

### Resetting stem diameters

The {meth}`~pyrealm.tmodel.TTree.reset_diameters` can be used to update an existing
{class}`~pyrealm.tmodel.TTree` object with a new set of diameters using the same
{class}`~pyrealm.constants.tmodel_const.TModelTraits` definition. Using
{meth}`~pyrealm.tmodel.TTree.reset_diameters` automatically resets any calculated growth
parameters: they will need to be recalculated for the new diameters.

```{code-cell} ipython3
tree1.reset_diameters(np.array([0.0001]))
print(tree1.height)
```

<!-- 
## The {func}`~pyrealm.tmodel.grow_ttree` function

The  {class}`~pyrealm.tmodel.TTree` class implements the calculation of the T Model
given diameter and GPP data. Using this calculate a time series just involves:

* Setting the diameter ({meth}`~pyrealm.tmodel.TTree.set_diameter`),
* calculating the annual growth ({meth}`~pyrealm.tmodel.TTree.calculate_growth`),
* adding the calculated {py:attr}`~pyrealm.tmodel.TTree.delta_d` to the current diameter
  and repeating.

This iteration is the main part of the {func}`~pyrealm.tmodel.grow_ttree`. The user
needs to provide initial stem diameters and then a time series of GPP values. The
function will return a `numpy` array containing any property of the
{class}`~pyrealm.tmodel.TTree` requested in the `outvars` argument of
{py:func}`~pyrealm.tmodel.grow_ttree`.

```{code-cell} ipython3
# Default traits
traits = tmodel.TModelTraits()
# A 1d array of 4 starting stem diameters
diams =  np.array([0.1, 0.3, 0.6, 0.9])
# An 2d array of GPP values for each stem for 100 years
gpp = np.array([[7] * 4] * 100)
# Run the simulation
years = np.arange(100)
values = tmodel.grow_ttree(gpp, diams, time_axis=0, traits=traits)
```

```{code-cell} ipython3
:tags: [hide-input]

fig, (ax1, ax2) = pyplot.subplots(1, 2, figsize=(10, 4))
ax1.plot(years, values[:, 0, 0], label='stem 0')
ax1.plot(years, values[:, 1, 0], label='stem 1')
ax1.plot(years, values[:, 2, 0], label='stem 2')
ax1.plot(years, values[:, 3, 0], label='stem 3')
ax1.set_title('Stem diameter')
ax1.set_xlabel('Years')
ax1.set_ylabel('Stem diameter (m)')
ax1.legend()
ax2.plot(years, values[:, 0, 1], label='stem 0')
ax2.plot(years, values[:, 1, 1], label='stem 1')
ax2.plot(years, values[:, 2, 1], label='stem 2')
ax2.plot(years, values[:, 3, 1], label='stem 3')
ax2.set_title('Height')
ax2.set_xlabel('Years')
ax2.set_ylabel('Height (m)')
ax2.legend()
pyplot.show()
``` -->
