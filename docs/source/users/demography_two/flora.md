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

# Plant Functional Types and Traits

```{admonition} Run this notebook
:class: hint

* Read the guide on setting up your computer to [run Jupyter
  notebooks](../getting_started.md)
* Download {nb-download}`this notebook<./flora.ipynb>` as a Jupyter notebook.

```

:::{admonition} Warning

This area of `pyrealm` is in active development and this notebook currently contains
notes and initial demonstration code.

:::

This page introduces the main components of the {mod}`~pyrealm.demography` module that
describe plant functional types (PFTs) and their traits.

```{code-cell} ipython3
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from pyrealm.demography_two.flora import Flora
```

## Plant traits

The table below shows the traits used to define the behaviour of different PFTs in
demographic simulations. These traits mostly consist of the parameters defined in the T
Model {cite}`Li:2014bc` to govern the allometric scaling and carbon allocation of trees,
but also include parameters for crown shape that follow the implementation developed in
the PlantFATE model {cite}`joshi:2022a`.

<!-- markdownlint-disable MD007 MD004 -->

:::{list-table}
:widths: 10 30
:header-rows: 1

* - Trait name
  - Description
* - `a_hd`
  - Initial slope of height-diameter relationship ($a$, -)
* - `ca_ratio`
  - Initial ratio of crown area to stem cross-sectional area ($c$, -)
* - `h_max`
  - Maximum tree height ($H_m$, m)
* - `rho_s`
  - Sapwood density ($\rho_s$, kg Cm-3)
* - `lai`
  - Leaf area index within the crown ($L$,  -)
* - `sla`
  - Specific leaf area ($\sigma$,  m2 kg-1 C)
* - `tau_f`
  - Foliage turnover time ($\tau_f$,years)
* - `tau_r`
  - Fine-root turnover time ($\tau_r$,  years)
* - `par_ext`
  - Extinction coefficient of photosynthetically active radiation (PAR) ($k$, -)
* - `yld`
  - Yield factor ($y$,  -)
* - `zeta`
  - Ratio of fine-root mass to foliage area ($\zeta$, kg C m-2)
* - `resp_r`
  - Fine-root specific respiration rate ($r_r$, year-1)
* - `resp_s`
  - Sapwood-specific respiration rate ($r_s$,  year-1)
* - `resp_f`
  - Foliage maintenance respiration fraction ($r_f$,  -)
* - `m`
  - Crown shape parameter ($m$, -)
* - `n`
  - Crown shape parameter ($n$, -)
* - `f_g`
  - Crown gap fraction ($f_g$, -)
* - `q_m`
  - Scaling factor to derive maximum crown radius from crown area.
* - `z_max_prop`
  - Proportion of stem height at which maximum crown radius is found.
:::

<!-- markdownlint-enable MD007 MD004 -->

+++

## The Flora class

The {class}`~pyrealm.demography_two.flora.Flora` class is used to create a set of PFTs
that will be used in a demographic simulation. It can be created directly by providing
a list of values for each trait: you must provide the same length list of values for
each trait but if you omit some traits then they will be automatically populated
from default values.

```{code-cell} ipython3
flora = Flora(name=["short", "medium", "tall"], h_max=[10, 20, 30])
```

The {meth}`~pyrealm.demography_two.Flora.to_dataframe` method exports the trait data as
a {class}`pandas.DataFrame`, making it easier to use for plotting or calculations
outside of `pyrealm`.

```{code-cell} ipython3
flora.to_dataframe().transpose()
```

You can also create a `Flora` instance using PFT data stored in a CSV file
formats.
