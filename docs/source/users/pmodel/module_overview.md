---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
language_info:
  codemirror_mode:
    name: ipython
    version: 3
  file_extension: .py
  mimetype: text/x-python
  name: python
  nbconvert_exporter: python
  pygments_lexer: ipython3
  version: 3.11.9
---

# The P Model module

The {mod}`~pyrealm.pmodel` module implements the P Model and extensions. The P Model
implements an eco-evolutionary optimality (EEO) theory  of plant productivity: that
plants optimise their behaviour to balance the costs and gains of photosynthesis.

The module provides implementations of the standard and subdaily forms of the P Model.
Much of the code and theory is shared between these implementations and is described in
the documentation for the [shared components](./shared_components/overview.md).

The two specific model implementations are then:

* The ['standard' P Model](pmodel_details/pmodel_overview)
  {cite:p}`Prentice:2014bc,Wang:2017go,Stocker:2020dh`.

* The [subdaily P Model](subdaily_details/subdaily_overview), extends the P Model to
  incorporate acclimation of photosynthetic pathways to changing environmental
  conditions {cite}`mengoli:2022a`. This improves the performance of the P Model at fine
  temporal resolutions, by accounting for lags in the adoption of optimal behaviour.

Note that the predictions of these models differ from instantaneous models of leaf level
ecophysiology, where the overall assimilation rate is found as the minimum of the
assimilation rate limited by carboxylation ($A_c$) and the assimilation rate limited by
electron transport $A_j$, given the internal $\ce{CO2}$ concentration ($c_i$). For more
details on fitting instantaneous responses see, for example, the R package
[`plantecophys`](https://remkoduursma.github.io/plantecophys/articles/Introduction_to_fitaci.html).

Instead, the P Model follows the coordination hypothesis
{cite}`haxeltine:1996a,maire:2012a`, which asserts that plants coordinate investment in
those two assimilation rates, such that $A_c = A_j$. Note that this coordination
requires time and so the standard P Model is typically used to make predictions from
average conditions on the scale of weeks to months. The subdaily P Model _also_ follows
the coordination hypothesis, but coordinates on the specific conditions encountered
during a daytime acclimation window and includes lags in how quickly the coordinated
investment adjusts to changing conditions.

In addition to the two P Model forms, this module also provides two extensions that
build on the P Model:

* The process of photosynthesis discriminates between the carbon isotopes occuring in
  air, leaving characteristic isotopic signatures. The [Isotopic
  discrimination](isotopic_discrimination) module estimates the isotopic signatures from
  different P Models.

* The relative advantages of the C3 and C4 photosynthetic pathways differ with
  environmental conditions. The [C3 / C4 plant competition](c3c4model) model uses the
  relative advantage to estimate the expected fraction of C4 plants in a community.
