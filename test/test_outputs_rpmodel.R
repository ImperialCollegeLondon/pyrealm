# This script takes a simple set of inputs to the rpmodel function in
# both scalar and vector form and extends that set to include the outputs
# of key functions and other intermediate variables to be validated using
# the pytest suite for the pmodel module

library(rpmodel)
library(yaml)

values <- read_yaml('test_inputs.yaml')

values <- within(values, {
       # calc_density_h2o
       dens_h20_sc <- calc_density_h2o(tc_sc, patm_sc)
       dens_h20_mx <- calc_density_h2o(tc_sc, patm_ar)  # NOTE using p_ar here
       dens_h20_ar <- calc_density_h2o(tc_ar, patm_ar)

       # calc_ftemp_arrh (using KattgeKnorr ha value)
       ftemp_arrh_sc <- calc_ftemp_arrh(tk_sc, KattgeKnorr_ha)
       ftemp_arrh_ar <- calc_ftemp_arrh(tk_ar, KattgeKnorr_ha)

       # calc_ftemp_inst_rd
       ftemp_inst_rd_sc <- calc_ftemp_inst_rd(tc_sc)
       ftemp_inst_rd_ar <- calc_ftemp_inst_rd(tc_ar)

       # calc_ftemp_inst_vcmax
       ftemp_inst_vcmax_sc <- calc_ftemp_inst_vcmax(tc_sc)
       ftemp_inst_vcmax_ar <- calc_ftemp_inst_vcmax(tc_ar)

       # calc_ftemp_kphio
       ftemp_kphio_c3_sc <- calc_ftemp_kphio(tc_sc)
       ftemp_kphio_c3_ar <- calc_ftemp_kphio(tc_ar)
       ftemp_kphio_c4_sc <- calc_ftemp_kphio(tc_sc, c4=TRUE)
       ftemp_kphio_c4_ar <- calc_ftemp_kphio(tc_ar, c4=TRUE)

       # calc_gammastar
       gammastar_sc <- calc_gammastar(tc_sc, patm_sc)
       gammastar_mx <- calc_gammastar(tc_ar, patm_sc)
       gammastar_ar <- calc_gammastar(tc_ar, patm_ar)

       # calc_kmm
       kmm_sc <- calc_kmm(tc_sc, patm_sc)
       kmm_mx <- calc_kmm(tc_ar, patm_sc)
       kmm_ar <- calc_kmm(tc_ar, patm_ar)

       # calc_soilmstress
       # This is **not vectorised** in rpmodel 1.0.6
       soilmstress_sc <- calc_soilmstress(soilm_sc, meanalpha_sc)
       soilmstress_mx <- mapply(calc_soilmstress, soilm_ar, meanalpha_sc)
       soilmstress_ar <- mapply(calc_soilmstress, soilm_ar, meanalpha_ar)

       # calc_viscosity_h2o
       viscosity_h2o_sc <- calc_viscosity_h2o(tc_sc, patm_sc)
       viscosity_h2o_mx <- calc_viscosity_h2o(tc_sc, patm_ar)  # NOTE using p_ar here
       viscosity_h2o_ar <- calc_viscosity_h2o(tc_ar, patm_ar)

       # ns_star
       visc_25 <- calc_viscosity_h2o(kTo, kPo)
       ns_star_sc <- viscosity_h2o_sc / visc_25
       ns_star_mx <- viscosity_h2o_mx / visc_25
       ns_star_ar <- viscosity_h2o_ar / visc_25

       # calc_patm
       patm_from_elev_sc <- calc_patm(elev_sc)
       patm_from_elev_ar <- calc_patm(elev_ar)

       # co2_to_ca
       # Not exported
       ca_sc <- rpmodel:::co2_to_ca(co2_sc, patm_sc)
       ca_mx <- rpmodel:::co2_to_ca(co2_ar, patm_sc)
       ca_ar <- rpmodel:::co2_to_ca(co2_ar, patm_ar)

       # rpmodel:::calc_chi_c4()
       # NOTE: this function doesn't do anything but return 1.0 scalars, but it
       # doesn't need to do anything else. There is no need to capture the input shape.
       optchi_c4 <- rpmodel:::calc_chi_c4()

       # rpmodel:::calc_optimal_chi
       optchi_p14_sc <- rpmodel:::calc_optimal_chi(kmm_sc, gammastar_sc, ns_star_sc,
                                                   ca_sc, vpd_sc, beta=146.0)
       optchi_p14_mx <- rpmodel:::calc_optimal_chi(kmm_ar, gammastar_sc, ns_star_ar,
                                                   ca_sc, vpd_sc, beta=146.0)
       optchi_p14_ar <- rpmodel:::calc_optimal_chi(kmm_ar, gammastar_ar, ns_star_ar,
                                                   ca_ar, vpd_ar, beta=146.0)
})

# Save values to YAML for use in python tests.
write_yaml(values, 'test_values_rpmodel.yaml', precision=10)
