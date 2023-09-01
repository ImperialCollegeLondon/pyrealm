# The `pyrealm` package

The `pyrealm` package provides a toolbox implementing some key models for estimating
plant productivity, growth and demography in Python 3. The outputs of different models
can be then easily fed into other models within `pyrealm` to allow productivity
estimates to be fed forward into estimation of net primary productivity, growth and
ultimately plant community demography.

The `pyrealm` package currently includes:

* The P Model for estimating optimal rates of plant photosynthesis given the balance
  between carbon capture and water loss. This includes recent extensions to incorporate
  the effects of water stress, slow acclimation processes, models of C3/C4 competition
  and carbon isotope fractionation.
* The T Model of the allocation of gross primary productivity to estimate net primary
  productivity and hence plant growth.

For more details, see the package website:
[https://pyrealm.readthedocs.io/](https://pyrealm.readthedocs.io/).

**TODO** Need to link here to a _roadmap_ for the package and therefore _create_ that
roadmap along with the a feature set to aim for in version 1.0.0.

## Using `pyrealm`

The `pyrealm` package requires Python 3.9 or greater and can be installed from
[PyPi](https://pypi.org/project/pyrealm/):

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
    co2=np.array([400]), patm=np.array([101325.0])
)

# Calculate the predictions of the P Model for a C3 plant
pmodel_c3 = PModel(env)

# Estimate the GPP from the model given the absorbed photosynthetically active light
pmodel_c3.estimate_productivity(fapar=1, ppfd=300)

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

## Developing `pyrealm`

If you are interested in contributing to the development of `pyrealm`, please read the
[guide for contributors](./CONTRIBUTING.md).

## Support and funding

Development of the `prealm` package has been supported by the following grants and
institutions:

* The [REALM project](https://prenticeclimategroup.wordpress.com/realm-team/), funded by
  an [ERC grant](https://cordis.europa.eu/project/id/787203) to Prof. Colin Prentice
  (Imperial College London).
* The [LEMONTREE project](https://research.reading.ac.uk/lemontree/), funded by Schmidt
  Futures through the [VESRI
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
  program funded by Schmidt Futures.
