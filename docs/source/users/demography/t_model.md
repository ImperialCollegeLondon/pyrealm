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
  version: 3.14.3
  mimetype: text/x-python
  codemirror_mode:
    name: ipython
    version: 3
  pygments_lexer: ipython3
  nbconvert_exporter: python
  file_extension: .py
---

# The T Model module

```{admonition} Run this notebook
:class: hint

* Read the guide on setting up your computer to [run Jupyter
  notebooks](../getting_started.md)
* Download {nb-download}`this notebook<./t_model.ipynb>` as a Jupyter notebook.

```

```{code-cell} ipython3
import warnings

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from pyrealm.core.experimental import ExperimentalFeatureWarning
from pyrealm.demography.flora import Flora
from pyrealm.demography.cohorts import create_cohorts, cohort_id_generator
from pyrealm.demography.tmodel import (
    StemAllocation,
    StemAllometry,
    GrowthIncrements,
    calculate_whole_crown_gpp,
)

warnings.filterwarnings(
    "ignore",
    category=ExperimentalFeatureWarning,
)
```

The T Model {cite}`Li:2014bc` defines both the allometry of trees and a carbon
allocation model for tree growth.

The [allometry of a stem](./allometry.md) is driven by the diameter at breast height
(DBH, metres) following a set of scaling relationships and defined [stem
traits](./flora.md) for its plant functional type (PFT).

The [carbon allocation for a stem](./carbon_allocation.md) partitions gross primary
productivity (GPP) in respiration, turnover, growth and efficiency losses. The diagram
below shows the allocation process:

* Net primary productivity (NPP) is GPP less respiration, but also subject to yield
  losses. The T Model includes terms for foliage, stem and fine root respiration.
* NPP is the carbon available for plant processes. The original T Model assumed all of
  the NPP went to biomass production, but the implementation in `pyrealm` allows users
  to modify NPP to apply other carbon costs, such as VOC emissions, root exudates or
  storage of non-structural carbohydrates.
* Biomass production is then the fraction of NPP used to produce plant biomass. It has
  to account for turnover costs (branch, foliage and fine root) and then any remaining
  biomass production can be allocated to growth. Growth is calculated as the incremental
  increase in DBH that accounts for the required increase in stem, foliage and fine root
  masses, given the stem allometry.

![GPP partition](./Allocation.png)
