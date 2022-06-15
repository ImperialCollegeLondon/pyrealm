"""Generate PModel test input values.

This file is used to generate a set of inputs to be fed into rpmodel to
create a test set of outputs for validation between the two implementations.

For validation, a mix of scalar and array inputs are created to check that
broadcasting works as intended.
"""

import numpy as np
import simplejson

# constants
KattgeKnorr_ha = 71513
kTo = 25.0
kPo = 101325.0
stocker_beta_c3 = 146
stocker_beta_c4 = 146 / 9
kphio = 0.05
c_molmass = 12.0107

# scalar versions
tc_sc = 20
tk_sc = tc_sc + 273.15
patm_sc = 101325
soilm_sc = 0.2
meanalpha_sc = 1
elev_sc = 1000
co2_sc = 400
vpd_sc = 1000
fapar_sc = 1
ppfd_sc = 300

# Create a sample from plausible ranges for variables for array inputs
n_samples = 100
rng = np.random.default_rng()

tc_ar = rng.uniform(-25, 50, size=n_samples)
tk_ar = np.round(tc_ar + 273.15, 5).tolist()
tc_ar = np.round(tc_ar, 5).tolist()
patm_ar = np.round(rng.uniform(50000, 108400, size=n_samples), 5).tolist()
elev_ar = np.round(rng.uniform(0, 3000, size=n_samples), 5).tolist()
vpd_ar = np.round(rng.uniform(0, 10000, size=n_samples), 5).tolist()
co2_ar = np.round(rng.uniform(280, 500, size=n_samples), 5).tolist()
fapar_ar = np.round(rng.uniform(0, 1, size=n_samples), 5).tolist()
ppfd_ar = np.round(rng.uniform(200, 400, size=n_samples), 5).tolist()
soilm_ar = np.round(rng.uniform(0.1, 0.7, size=n_samples), 5).tolist()
meanalpha_ar = np.round(rng.uniform(0.2, 1.0, size=n_samples), 5).tolist()


out_dict = dict(
    # constants
    KattgeKnorr_ha=KattgeKnorr_ha,
    kTo=kTo,
    kPo=kPo,
    stocker_beta_c3=stocker_beta_c3,
    stocker_beta_c4=stocker_beta_c4,
    kphio=kphio,
    c_molmass=c_molmass,
    # scalar forcings
    tc_sc=tc_sc,
    tk_sc=tk_sc,
    patm_sc=patm_sc,
    elev_sc=elev_sc,
    co2_sc=co2_sc,
    vpd_sc=vpd_sc,
    fapar_sc=fapar_sc,
    ppfd_sc=ppfd_sc,
    soilm_sc=soilm_sc,
    meanalpha_sc=meanalpha_sc,
    # array forcings
    tc_ar=tc_ar,
    tk_ar=tk_ar,
    patm_ar=patm_ar,
    elev_ar=elev_ar,
    co2_ar=co2_ar,
    vpd_ar=vpd_ar,
    soilm_ar=soilm_ar,
    meanalpha_ar=meanalpha_ar,
    fapar_ar=fapar_ar,
    ppfd_ar=ppfd_ar,
    # shape_error
    shape_error=[1, 1, 1],
)

with open("test_inputs.json", "w") as outf:
    simplejson.dump(out_dict, outf, indent="  ")
