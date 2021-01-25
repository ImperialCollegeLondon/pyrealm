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


# Scaling to absolute values

Calculations in the P-model are calculated  relative to a unit of absorbed
photosynthetically active radiation. To calculate absolute values values for
both the fraction of absorbed photosynthetically active radiation (`fapar`) and
the photosynthetic photon flux density (`ppfd`) must be provided to calculate
an amount of absorbed light ($I_{abs}$).

## Absorbed photosynthetically active radiation

The value of $I_{abs} is simply:

.. math::

    I_{abs} = \text{FAPAR} \cdot \text{PPFD}

Note that the units of PPFD determine the units of outputs: if PPFD is
in $\text{mol} m^{-2} \text{month}^{-1}, then respective output variables 
are scaled per month.

## Gross primary productivity

