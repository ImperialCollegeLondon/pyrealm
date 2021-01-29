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

# Parameterisation

The models presented in this package rely on a relatively large number of
underlying parameters. In order to simplify usage, this package discriminates
between:

Function arguments:
    The values that a user is most likely to want to alter.

Parameterisation:
    A set of underlying values that are likely to be constant for a particular
    study. Many of these are true constants – such as the universal gas
    constant $R=8.3145 J \cdot K^{-1} \cdot mol^{-1}$. However many others
    are estimates derived from the literature and a user might want to update
    a value or explore sensitivity to variation.

For this reason, the package defines a set of parameters that are automatically
loaded ({mod}`pyrealm.params`) into the global dictionary
{const}`pyrealm.params.PARAM`. The values in this dictionary can be edited by
users to change the default values. Note that many of these variables are
defined using a standard reference temperature of 25.0 °C (`PARAM.k.pTo`).

```{eval-rst}
.. automodule:: pyrealm.params
    :members: PARAM
```

## `data/params.yaml`

The values in `PARAM` are loaded from the package data file `data/params.yaml`
and the contents of this are shown below:

```{eval-rst}
.. include:: ../pyrealm/data/params.yaml
    :code: yaml
```
