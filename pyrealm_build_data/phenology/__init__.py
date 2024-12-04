"""This submodule provides regression test data from the initial implentation of LAI
phenology calculations, provided by Boya Zhou. The original provided data consisted of
two files:

* **FLX_DE-Gri_FLUXNET2015_FULLSET_HH_2004-2014_1-4.csv**: This contained the complete
  FluxNET data set for the 'DE-Gri' site at half hourly resolution and was around 350
  MB. It has been reduced to remove fields not used in the calculations to reduce file
  size, creating the file `` FLX_DE-Gri_FLUXNET2015_phenology_inputs.csv``. The file
  also includes some half-hourly predictions from the subdaily P Model implementation to
  validate the subdaily calculations.

* **DE_Gri_Grassland_example.xlsx**: This contained daily calculations from the half
  hourly data, along with predictions of various phenological variables. This file has
  also been reduced to remove un-needed fields, creating the file
  ``DE_Gri_Grassland_example_subset.csv``.

* In addition, both files contained required data that is constant across all
  observations, such as the site coordinates and aridity index. These constants have
  been extracted into the file ``DE-GRI_site_data.json``.

"""  # noqa: D205, D415
