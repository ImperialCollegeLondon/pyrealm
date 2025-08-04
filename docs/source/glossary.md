---
jupyter:
  jupytext:
    cell_metadata_filter: all,-trusted
    main_language: python
    notebook_metadata_filter: settings,mystnb,language_info,ve_data_science,-jupytext.text_representation.jupytext_version
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
---

# Glossary

:::{glossary}

Allometry
  Allometric relationships describe the connection between the size of an organism and
  other aspects of its shape. In `pyrealm`, the diameter at breast height (DBH) is used
  as the principle measure of the size of a tree and allometric relationships then
  capture how other stem traits such as height and canopy size scale with DBH.

AET
  Actual evapotranspiration (AET) measures the amount of water actually evaporated from
  the land surface or transpired through leaves. When water is not limiting, then the
  AET will be equal to the potential evapotranspiration (PET), but AET will decrease
  as conditions become more arid. See also aridity index (AI).

AI
  The aridity index (AI) is a measure of how dry the climate at a sampling location and
  is calculated as the ratio of potential evapotranspiration over total precipitation
  (PET/P). It is taken as a climatalogical value over a long time period, typically at
  least 20 years, and arid locations have large values (high PET over low
  precipitation). Note that AI is also sometimes calculated as the inverse of this
  (P/PET), giving small values for arid locations.

LAI
  The leaf area index (LAI or $L$) is a measure of the amount of leaf area present in
  given area of canopy. It is typically measured in m2 of leaf area per m2 of canopy and
  can be thought of as measure of the thickness of the canopy. As the LAI increases, the
  ability of the canopy to absorb light increases via the Beer Lambert law, which
  calculates FAPAR as $1 - e^{-kL}$, where $k$ is an absorption coefficient.

FAPAR
  The fraction of absorbed photosynthetically active radiation (FAPAR or
  $\text{fAPAR}$) is a measure of the fraction of the of downwelling radiation usable by
  plants that is actually absorbed. It is sometimes calculated from remotely sensed data
  and can also be calculated from the leaf area index (LAI).

DBH
  The diameter at breast height (DBH or $D$) is a measure of the size of a tree stem: the
  diameter around the stem measured at a fixed height above the ground. Although the
  height is typically fixed within studies, there is no single global definition: it is
  most commonly at 130 cm but does vary {cite}`brokaw:2000a`.

GPP
  Gross primary productivity (GPP) is a measure of the total carbon assimilation by
  plants over a given time period before accounting for respiration and any maintenance
  or turnover costs for plant tissues. The assimilation remaining after accounting for
  respiration and other costs is the net primary productivity (NPP). The P Model classes
  in `pyrealm` return GPP measured in  µg C m-2 s-1.

PFT
  A plant functional type (PFT) defines a particular plant growth form. Within
  `pyrealm`, a PFT is defined using the set of traits defined in the T Model and that
  govern the growth form of a tree along with its wood density and turnover and
  maintenance costs.

PPFD
  The photosynthetic photon flux density (PPFD) is the rate at which photosynthetically
  active photons (wavelength of 400-700nm) reach a plant canopy. It is typically
  measured in µmol m-2 s-1.

:::
