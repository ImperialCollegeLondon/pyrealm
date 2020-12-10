library(rpmodel)
library(yaml)

inputs <- read_yaml('test_inputs.yaml')
outputs <- list()

inputs <- within(inputs, {
       # calc_density_h2o
       outputs$dens_h20_sc <- calc_density_h2o(tc, patm)
       outputs$dens_h20_sc_ar <- calc_density_h2o(tc, patm_mat)  # NOTE using p_mat here
       outputs$dens_h20_ar <- calc_density_h2o(tc_mat, patm_mat)

       # calc_ftemp_arrh (using KattgeKnorr ha value)
       outputs$ftemp_arrh_sc <- calc_ftemp_arrh(tk, KattgeKnorr_ha)
       outputs$ftemp_arrh_ar <- calc_ftemp_arrh(tk_mat, KattgeKnorr_ha)

       # calc_ftemp_inst_rd
       outputs$ftemp_inst_rd_sc <- calc_ftemp_inst_rd(tc)
       outputs$ftemp_inst_rd_ar <- calc_ftemp_inst_rd(tc_mat)

       # calc_ftemp_inst_vcmax
       outputs$ftemp_inst_vcmax_sc <- calc_ftemp_inst_vcmax(tc)
       outputs$ftemp_inst_vcmax_ar <- calc_ftemp_inst_vcmax(tc_mat)

       # calc_ftemp_kphio
       outputs$ftemp_kphio_c3_sc <- calc_ftemp_kphio(tc)
       outputs$ftemp_kphio_c3_ar <- calc_ftemp_kphio(tc_mat)
       outputs$ftemp_kphio_c4_sc <- calc_ftemp_kphio(tc, c4=TRUE)
       outputs$ftemp_kphio_c4_ar <- calc_ftemp_kphio(tc_mat, c4=TRUE)

       # calc_gammastar
       outputs$gammastar_sc <- calc_gammastar(tc, patm)
       outputs$gammastar_sc_ar <- calc_gammastar(tc_mat, patm)
       outputs$gammastar_ar <- calc_gammastar(tc_mat, patm_mat)

       # calc_kmm
       outputs$kmm_sc <- calc_kmm(tc, patm)
       outputs$kmm_sc_ar <- calc_kmm(tc_mat, patm)
       outputs$kmm_ar <- calc_kmm(tc_mat, patm_mat)

       # calc_soilmstress
       # This is **not vectorised** in rpmodel 1.0.6
       outputs$soilmstress_sc <- calc_soilmstress(soilm, meanalpha)
       outputs$soilmstress_sc_ar <- mapply(calc_soilmstress, soilm_mat, meanalpha)
       outputs$soilmstress_ar <- mapply(calc_soilmstress, soilm_mat, meanalpha_mat)

       # calc_viscosity_h2o
       outputs$viscosity_h2o_sc <- calc_viscosity_h2o(tc, patm)
       outputs$viscosity_h2o_sc_ar <- calc_viscosity_h2o(tc, patm_mat)  # NOTE using p_mat here
       outputs$viscosity_h2o_ar <- calc_viscosity_h2o(tc_mat, patm_mat)

       # calc_patm
       outputs$patm_sc <- calc_patm(elev)
       outputs$patm_ar <- calc_patm(elev_mat)

       # co2_to_ca
       # Not exported
       outputs$co2_to_ca_sc <- rpmodel:::co2_to_ca(co2, patm)
       outputs$co2_to_ca_sc_ar <- rpmodel:::co2_to_ca(co2_mat, patm)
       outputs$co2_to_ca_ar <- rpmodel:::co2_to_ca(co2_mat, patm_mat)

       # rpmodel:::calc_chi_c4()
       # NOTE: this function doesn't do anything but return 1.0 scalars, but it
       # doesn't need to do anything else. There is no need to capture the input shape.
       outputs$optchi_c4 <- rpmodel:::calc_chi_c4()

       # rpmodel:::calc_optimal_chi
       outputs$optchi_p14_sc <- rpmodel:::calc_optimal_chi(kmm, gammastar, ns_star, ca, vpd, beta=146.0)
       outputs$optchi_p14_sc_ar <- rpmodel:::calc_optimal_chi(kmm_mat, gammastar, ns_star_mat, ca, vpd, beta=146.0)
       outputs$optchi_p14_ar <- rpmodel:::calc_optimal_chi(kmm_mat, gammastar_mat, ns_star_mat, ca_mat, vpd_mat, beta=146.0)
})

# Save outputs to YAML for use in python tests.
write_yaml(inputs$outputs, 'test_outputs_rpmodel.yaml', precision=10)
