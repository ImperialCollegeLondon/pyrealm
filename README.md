
<!-- markdownlint-disable-next-line MD041 MD033 -->
<img
  alt="The pyrealm logo: a green leaf over the shining sun."
  src="/docs/source/_static/images/pyrealm_logo.png" width=50%>

---

[![PyPI - Version](https://img.shields.io/pypi/v/pyrealm)](https://pypi.org/project/pyrealm/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyrealm)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8366847.svg)](https://doi.org/10.5281/zenodo.8366847)
[![Documentation
Status](https://readthedocs.org/projects/pyrealm/badge/?version=latest)](https://pyrealm.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/ImperialCollegeLondon/pyrealm/branch/develop/graph/badge.svg)](https://codecov.io/gh/ImperialCollegeLondon/pyrealm)
[![Test and
build](https://github.com/ImperialCollegeLondon/pyrealm/actions/workflows/pyrealm_ci.yaml/badge.svg?branch=develop)](https://github.com/ImperialCollegeLondon/pyrealm/actions/workflows/pyrealm_ci.yaml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/ImperialCollegeLondon/pyrealm/develop.svg)](https://results.pre-commit.ci/latest/github/ImperialCollegeLondon/pyrealm/develop)
[![SPEC 0 — Minimum Supported Dependencies](https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0000/)

The `pyrealm` package provides a toolbox implementing some key models for estimating
plant productivity, growth and demography in Python. The outputs of different models
can be then easily fed into other models within `pyrealm` to allow productivity
estimates to be fed forward into estimation of net primary productivity, growth and
ultimately plant community demography.

The `pyrealm` package currently includes:

* The P Model for estimating optimal rates of plant photosynthesis given the balance
  between carbon capture and water loss. This includes recent extensions to incorporate
  the effects of water stress, slow acclimation processes, models of C3/C4 competition
  and carbon isotope fractionation.
* The SPLASH model for calculating soil moisture and actual evapotranspiration.
* Initial components of a demography module for modelling plant functional types,
  cohorts, communities and canopies, including allocation calculations using the T
  Model.
* A suite of core physics functions and other utilities used to support the modules
  above.

For more details, see the package website:
[https://pyrealm.readthedocs.io/](https://pyrealm.readthedocs.io/).

## Version 2.0.0 development

New functionality being implemented after version 1.0.0 has lead to some immediate
breaking changes in the API, for example in the handling of quantum yield settings in
the class signatures for the `PModel` and `SubdailyPModel`. As the package uses
[semantic versioning](https://semver.org/), these changes to the API require that new
releases be made under a new major version.

We will be publishing a series of "release candidates" of the 2.0.0 package. These will
be used to identify issues with the current API and try to stabilise a new API. The
content of version 2.0.0 is not yet finalised, so these release candidates may also add
new functionality.

We recommend that you update to the most recent release candidate of version 2.0.0. The
documentation now includes a [migration
guide](https://pyrealm.readthedocs.io/en/develop/users/versions.html) to help update
existing code.

## Using `pyrealm`

The `pyrealm` package requires Python 3 and the currently supported Python versions are:
3.10, 3.11 and 3.12. We make released package versions available via
[PyPi](https://pypi.org/project/pyrealm/) and also generate DOIs for each release via
[Zenodo](https://doi.org/10.5281/zenodo.8366847). You can install the most recent
release using `pip`:

```sh
pip install pyrealm
```

You can now get started using `pyrealm`. For example, to calculate the estimated gross
primary productivity of a C3 plant in a location, start a Python interpreter, using
`python`, `python3` or `ipython` depending on your installation, and run:

```python
import numpy as np
from pyrealm.pmodel import PModelEnvironment, PModel

# Calculate the photosynthetic environment given the conditions
env = PModelEnvironment(
    tc=np.array([20]), vpd=np.array([1000]),
    co2=np.array([400]), patm=np.array([101325.0],
    fapar=1, ppfd=300)
)

# Calculate the predictions of the P Model for a C3 plant
pmodel_c3 = PModel(env)

# Report the GPP in micrograms of carbon per m2 per second.
pmodel_c3.gpp
```

This should give the following output:

```python
array([76.42544948])
```

The package website provides worked examples of using `pyrealm`, for example to:

* [fit the P
  Model](https://pyrealm.readthedocs.io/en/latest/users/pmodel/pmodel_details/worked_examples.html),
* [include acclimation in estimating light use
  efficiency](https://pyrealm.readthedocs.io/en/latest/users/pmodel/subdaily_details/worked_example.html)
  , and
* [estimate C3/C4
  competition](https://pyrealm.readthedocs.io/en/latest/users/pmodel/c3c4model.html#worked-example).

These worked examples also show how `pyrealm` can be used within Python scripts or
Jupyter notebooks and how to use `pyrealm` with large datasets loaded using
[`numpy`](https://numpy.org/) or [`xarray`](https://docs.xarray.dev/en/stable/) with
`pyrealm` classes and functions.

## Citing `pyrealm`

The `pyrealm` repository can be cited following the information in the [citation
file](./CITATION.cff). If you are using `pyrealm` in research, it is better to cite the
DOI of the specific release from [Zenodo](https://doi.org/10.5281/zenodo.8366847).

## Developing `pyrealm`

If you are interested in contributing to the development of `pyrealm`, please read the
[guide for contributors](./CONTRIBUTING.md). Please do also read the [code of
conduct](./CODE_OF_CONDUCT.md) for contributing to this project.

## Support and funding

Development of the `pyrealm` package has been supported by the following grants and
institutions:

* The [REALM project](https://prenticeclimategroup.wordpress.com/realm-team/), funded by
  an [ERC grant](https://cordis.europa.eu/project/id/787203) to Prof. Colin Prentice
  (Imperial College London).
* The [LEMONTREE project](https://research.reading.ac.uk/lemontree/), funded by Schmidt
  Sciences through the [VESRI
  programme](https://www.schmidtfutures.com/our-work/virtual-earth-system-research-institute-vesri/)
  to support an international research team lead by Prof. Sandy Harrison (University of
  Reading).
* The [Virtual Rainforest project](https://pyrealm.readthedocs.io/), funded by a
  Distinguished Scientist award from the [NOMIS
  Foundation](https://nomisfoundation.ch/research-projects/a-virtual-rainforest-for-understanding-the-stability-resilience-and-sustainability-of-complex-ecosystems/)
  to Prof. Robert Ewers (Imperial College London)
* Research software engineering support from the [Institute of Computing for Climate
  Science](https://iccs.cam.ac.uk/) at the University of Cambridge, through the [Virtual
  Institute for Scientific
  Software](https://www.schmidtfutures.com/our-work/virtual-institute-for-scientific-software/)
  program funded by Schmidt Sciences.
