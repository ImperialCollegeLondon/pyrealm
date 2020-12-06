.. _parameterisation:

Parameterisation
================

The models presented in this package rely on a relatively large number of
underlying parameters. In order to simplify usage, this package discriminates
between:

Function arguments:
    The values that a user is most likely to want to alter.

Parameterisation:
    A set of underlying values that are likely to be constant for a particular
    study. Many of these are true constants – such as the universal gas
    constant :math:`R=8.3145 J \cdot K^{-1} \cdot mol^{-1}`. However many others
    are estimates derived from the literature and a user might want to update
    a value or explore sensitivity to variation.

For this reason, the package defines a set of parameters that are automatically
loaded (:mod:`pypmodel.params`) into the global :mod:`pypmodel.params.PARAM`
variable. This can be edited by users to change these underlying values.

Note that many of these variables are defined using a standard reference
temperature of 25.0 °C (`PARAM.k.pTo`).


The values in `PARAM` are loaded from the package data file `data/params.yaml`
and the contents of this are shown below:


`data/params.yaml`
------------------

.. include:: ../pypmodel/data/params.yaml
    :code: yaml


.. automodule:: pypmodel.params
    :members:
