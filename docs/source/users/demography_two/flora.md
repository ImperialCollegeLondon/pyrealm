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
  name: python
  version: 3.12.3
  mimetype: text/x-python
  codemirror_mode:
    name: ipython
    version: 3
  pygments_lexer: ipython3
  nbconvert_exporter: python
  file_extension: .py
---

# Plant functional types and cohorts

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

This page introduces the main components of the {mod}`~pyrealm.demography` module that:

* describe plant functional types (PFTs) and their traits
* define size-structured cohorts as a number of individuals from a specific PFT with a
  given diameter at breast height (DBH).

```{code-cell} ipython3
import numpy as np

from pyrealm.demography_two.flora import Flora
from pyrealm.demography_two.cohorts import Cohorts
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
flora
```

The {meth}`~pyrealm.demography_two.Flora.to_dataframe` method exports the trait data as
a {class}`pandas.DataFrame`, making it easier to use for plotting or calculations
outside of `pyrealm`.

```{code-cell} ipython3
flora.to_dataframe().transpose()
```

You can also create a `Flora` instance using PFT data stored in a [CSV
file](./pfts.csv). Note that this CSV only provides some of the PFT traits, you can use
`Flora.from_csv("pfts.csv", strict=True)` to require that the file provides all the
traits.

```{code-cell} ipython3
flora_from_csv = Flora.from_csv("pfts.csv")
flora_from_csv
```

## Plant Cohorts

The {class}`~pyrealm.demography_two.cohorts.Cohorts` object provides a class to describe
size structured cohorts of plants. A cohort is simply a number of individuals of a given
PFT with size specified as diameter at breast height. The `Cohorts` object validates the
cohort data and pairs each cohort up with the appropriate set of traits from a provided
`Flora`.

The `Cohorts` object automatically assigns a unique cohort ID  to each cohort. The
details of these ID values can be controlled through the `cid_generator` argument (see
{class}`~pyrealm.demography_two.cohorts.Cohorts`)

```{code-cell} ipython3
# Create a simple community with three cohorts
# - 15 saplings of the short PFT
# - 5 larger stems of the short PFT
# - 2 large stems of tall PFT

cohorts = Cohorts(
    dbh_value=np.array([0.02, 0.20, 0.5]),
    n_individuals=np.array([15, 5, 2]),
    pft_name=np.array(["short", "short", "tall"]),
    flora=flora,
)

cohorts
```

The `Cohorts` object contains a dataframe of the provided cohort data, extended to
include the functional traits associated with the cohort PFT.

```{code-cell} ipython3
cohorts.cohorts[["cohort_id", "n_individuals", "dbh_value", "pft_name", "h_max"]]
```

```{code-cell} ipython3

```
