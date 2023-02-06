---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Constants

The models presented in this package rely on a relatively large number of underlying
constants. In order to keep the argument lists of functions and classes as simple as
possible, the `pyrealm` package discriminates between two kinds of variables.

1. **Arguments**: Class and function arguments cover the variables that a user
is likely to want to vary within a particular study, such as temperature or
primary productivity.

2. **Model Constants**: These are the underlying values that are likely to be constant
for a particular study. These may be true constants (such as the universal gas constant
$R$) but can also be experimental estimates of coefficients for functions describing
plant physiology or geometry, which a user might want to alter to update with a new
estimate or explore sensitivity to variation.

For this reason, the `pyrealm` package provides data classes that contain sets
of default model constants:

* {class}`~pyrealm.constants.pmodel_const.PModelConst` for the
  {class}`~pyrealm.pmodel.pmodel.PModel`
* {class}`~pyrealm.constants.tmodel_const.TModelTraits` for the
  {class}`~pyrealm.tmodel.TModel`

## Creating constant class instances

These can be used to generate the default set of model parameters:

```{code-cell} python
from pyrealm.constants import PModelConst, TModelTraits

ppar = PModelConst()
ttrt = TModelTraits()

print(ppar)
print(ttrt)
```

And individual values can be altered using the parameter arguments:

```{code-cell} python
# Simulate the P Model under the moon's gravity...
ppar_moon = PModelConst(k_G = 1.62)
# ... allowing a much greater maximum height
ttrt_moon = TModelTraits(h_max=200)

print(ppar_moon.k_G)
print(ttrt_moon.h_max)
```

In order to ensure that a set of parameters cannot change while models are being run,
instances of these parameter classes are **frozen**. You cannot  edit an existing
instance and will need to create a new instance to use different parameters.

```{code-cell} python
:tags: ["raises-exception"]
ppar_moon.k_G = 9.80665
```

## Exporting and reloading parameter sets

All parameter classes inherit methods from the base
{class}`pyrealm.constants.ConstantsClass` that provide bulk import and export of
parameter settings to dictionaries and to JSON formatted files.

```{eval-rst}
.. autoclass:: pyrealm.constants.base.ConstantsClass
    :members: from_dict, to_dict, from_json, to_json
```

The code below shows these methods working. First, a trait definition in a JSON
file is read into a dictionary:

```{code-cell} python
import json
import pprint
trt_dict = json.load(open('files/traits.json', 'r'))
pprint.pprint(trt_dict)
```

That dictionary can  then be used to create a TModelTraits instance using
the {meth}`~pyrealm.constants.base.ConstantsClass.from_dict` method. The
{meth}`~pyrealm.constants.base.ConstantsClass.from_json` method allows this to
be done more directly and the resulting instances are identical.

```{code-cell} python
traits1 = TModelTraits.from_dict(trt_dict)
traits2 = TModelTraits.from_json('files/traits.json')

print(traits1)
print(traits2)

traits1 == traits2
```

## P Model parameters

```{eval-rst}
.. autoclass:: pyrealm.constants.pmodel_const.PModelConst
```

### Dictionary of default values for PModel parameters

```{code-cell} python
const = PModelConst()
pprint.pprint(const.to_dict())
```

## T Model traits

```{eval-rst}
.. autoclass:: pyrealm.constants.tmodel_const.TModelTraits
```

### Dictionary of default values for TModel traits

```{code-cell} python
traits = TModelTraits()
pprint.pprint(traits.to_dict())
```

## Hygro parameters

```{eval-rst}
.. autoclass:: pyrealm.constants.hygro_const.HygroConst
    :members:
```
