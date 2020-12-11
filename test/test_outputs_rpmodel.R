# This script takes a simple set of inputs to the rpmodel function in
# both scalar and array form and extends that set to include the outputs
# of key functions and other intermediate variables to be validated using
# the pytest suite for the pmodel module

library(rpmodel)
library(yaml)

values <- read_yaml('test_inputs.yaml')

values <- within(values, {
       # calc_density_h2o
       dens_h20_sc <- calc_density_h2o(tc_sc, patm_sc)
       dens_h20_mx <- calc_density_h2o(tc_ar, patm_sc)
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
       viscosity_h2o_mx <- calc_viscosity_h2o(tc_ar, patm_sc)
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

# CalcVUEVcmax tests

sm <- c('sm-off', 'sm-on')

ft <- c('fkphio-off', 'fkphio-on')

lue_method <- c('wang17', 'smith19', 'none')

optchi  <- list(sc = list(kmm=values$kmm_sc,
                          gammastar=values$gammastar_sc,
                          ns_star=values$ns_star_sc,
                          ca=values$ca_sc,
                          vpd=values$vpd_sc,
                          beta=146.0),
                ar = list(kmm=values$kmm_ar,
                          gammastar=values$gammastar_ar,
                          ns_star=values$ns_star_ar,
                          ca=values$ca_ar,
                          vpd=values$vpd_ar,
                          beta=146.0))

# Needs to match to (reverse) ordering of pytest.mark.parametrise
# variables in pytesting.
luevcmax_c3 <- expand.grid(oc=names(optchi),
                           lm=lue_method,
                           ft=ft,
                           sm=sm,
                           stringsAsFactors=FALSE)

for (rw in seq(nrow(luevcmax_c3))){

    inputs <- as.list(luevcmax_c3[rw,])

    if (inputs$ft == 'fkphio-off'){
        ftemp_kphio <- 1.0
    } else if (inputs$oc == 'sc'){
        ftemp_kphio <- calc_ftemp_kphio(tc=values$tc_sc, c4=FALSE)
    } else {
        ftemp_kphio <- calc_ftemp_kphio(tc=values$tc_ar, c4=FALSE)
    }

    # Optimal Chi
    optchi_out <- do.call(rpmodel:::calc_optimal_chi, optchi[[inputs$oc]])

    # Soilmstress
    if (inputs$sm == 'sm-off'){
        soilmstress <- 1.0
    } else {
        soilmstress <- calc_soilmstress(soilm=values$soilm_sc,
                                        meanalpha=values$meanalpha_sc)
    }

    test_name <- paste('c3', paste(inputs, collapse='-'), sep='-')
    values[[test_name]] <- switch(inputs$lm,
                  'wang17' = rpmodel:::calc_lue_vcmax_wang17(optchi_out, kphio=0.05,
                                ftemp_kphio=ftemp_kphio, soilmstress=soilmstress,
                                c_molmass=12.0107),
                  'smith19' = rpmodel:::calc_lue_vcmax_smith19(optchi_out, kphio=0.05,
                                ftemp_kphio=ftemp_kphio, soilmstress=soilmstress,
                                c_molmass=12.0107),
                  'none' = rpmodel:::calc_lue_vcmax_none(optchi_out, kphio=0.05,
                                ftemp_kphio=ftemp_kphio, soilmstress=soilmstress,
                                c_molmass=12.0107))
}


# Needs to match to (reverse) ordering of pytest.mark.parametrise
# variables in pytesting.
luevcmax_c4 <- expand.grid(ft=ft,
                           sm=sm,
                           stringsAsFactors=FALSE)

for (rw in seq(nrow(luevcmax_c4))){

    inputs <- as.list(luevcmax_c4[rw,])

    if (inputs$ft == 'fkphio-off'){
        ftemp_kphio <- 1.0
    } else {
        ftemp_kphio <- calc_ftemp_kphio(tc=values$tc_sc, c4=TRUE)
    }

    # Optimal Chi
    optchi_out <- rpmodel:::calc_chi_c4()

    # Soilmstress
    if (inputs$sm == 'sm-off'){
        soilmstress <- 1.0
    } else {
        soilmstress <- calc_soilmstress(soilm=values$soilm_sc,
                                        meanalpha=values$meanalpha_sc)
    }

    test_name <- paste('c4', paste(inputs, collapse='-'), sep='-')
    values[[test_name]] <- rpmodel:::calc_lue_vcmax_c4(kphio=0.05,
                                ftemp_kphio=ftemp_kphio, soilmstress=soilmstress,
                                c_molmass=12.0107)
}


# Save values to YAML for use in python tests.
write_yaml(values, 'test_outputs_rpmodel.yaml', precision=10)
