# This script takes a simple set of inputs to the rpmodel function in
# both scalar and array form and extends that set to include the outputs
# of key functions and other intermediate variables to be validated using
# the pytest suite for the pmodel module

library(rpmodel)
library(jsonlite)

values <- fromJSON(file("test_inputs.json"))

# Record the version details of rpmodel
pkg_desc <-  packageDescription('rpmodel')

if (is.null(pkg_desc$Repository)) {
    # Github install
    pkg_desc <- pkg_desc[c('Version', 'GithubSHA1')]
} else {
    # CRAN install
    pkg_desc <- pkg_desc[c('Version', 'Repository')]
}

values$rpmodel_version <- pkg_desc

values <- within(values, {
       # density_h2o
       dens_h20_sc <- density_h2o(tc_sc, patm_sc)
       dens_h20_mx <- density_h2o(tc_ar, patm_sc)
       dens_h20_ar <- density_h2o(tc_ar, patm_ar)

       # ftemp_arrh (using KattgeKnorr ha value)
       ftemp_arrh_sc <- ftemp_arrh(tk_sc, KattgeKnorr_ha)
       ftemp_arrh_ar <- ftemp_arrh(tk_ar, KattgeKnorr_ha)

       # ftemp_inst_rd
       ftemp_inst_rd_sc <- ftemp_inst_rd(tc_sc)
       ftemp_inst_rd_ar <- ftemp_inst_rd(tc_ar)

       # ftemp_inst_vcmax
       ftemp_inst_vcmax_sc <- ftemp_inst_vcmax(tc_sc)
       ftemp_inst_vcmax_ar <- ftemp_inst_vcmax(tc_ar)

       # ftemp_kphio
       ftemp_kphio_c3_sc <- ftemp_kphio(tc_sc)
       ftemp_kphio_c3_ar <- ftemp_kphio(tc_ar)
       ftemp_kphio_c4_sc <- ftemp_kphio(tc_sc, c4 = TRUE)
       ftemp_kphio_c4_ar <- ftemp_kphio(tc_ar, c4 = TRUE)

       # gammastar
       gammastar_sc <- calc_gammastar(tc_sc, patm_sc)
       gammastar_mx <- calc_gammastar(tc_ar, patm_sc)
       gammastar_ar <- calc_gammastar(tc_ar, patm_ar)

       # kmm
       kmm_sc <- calc_kmm(tc_sc, patm_sc)
       kmm_mx <- calc_kmm(tc_ar, patm_sc)
       kmm_ar <- calc_kmm(tc_ar, patm_ar)

       # soilmstress
       # Bug for mixed inputs in rpmodel:
       # https://github.com/computationales/rpmodel/issues/16
       # Note that calc_soilmstress retains the outdated default b=0.685
       bpar_soilm <- 0.733
       soilmstress_sc <- calc_soilmstress(soilm_sc, meanalpha_sc,
                                          bpar_soilm = bpar_soilm)
       soilmstress_mx <- mapply(calc_soilmstress, soilm_ar, meanalpha_sc,
                                bpar_soilm = bpar_soilm)
       soilmstress_ar <- calc_soilmstress(soilm_ar, meanalpha_ar,
                                          bpar_soilm = bpar_soilm)

       # viscosity_h2o
       viscosity_h2o_sc <- viscosity_h2o(tc_sc, patm_sc)
       viscosity_h2o_mx <- viscosity_h2o(tc_ar, patm_sc)
       viscosity_h2o_ar <- viscosity_h2o(tc_ar, patm_ar)

       # ns_star
       visc_25 <- viscosity_h2o(kTo, kPo)
       ns_star_sc <- viscosity_h2o_sc / visc_25
       ns_star_mx <- viscosity_h2o_mx / visc_25
       ns_star_ar <- viscosity_h2o_ar / visc_25

       # patm
       patm_from_elev_sc <- calc_patm(elev_sc)
       patm_from_elev_ar <- calc_patm(elev_ar)

       # co2_to_ca
       ca_sc <- co2_to_ca(co2_sc, patm_sc)
       ca_mx <- co2_to_ca(co2_ar, patm_sc)
       ca_ar <- co2_to_ca(co2_ar, patm_ar)

       # rpmodel:::optimal_chi
       optchi_p14_sc_c3 <- rpmodel:::optimal_chi(kmm_sc, gammastar_sc,
                                                 ns_star_sc, ca_sc, vpd_sc,
                                                 beta = stocker_beta_c3,
                                                 c4 = FALSE)
       # The mx version is odd - the pyrealm test uses tc_ar and then
       # scalars for the other inputs to PModelEnvironment, so gammastar
       # ns_star and kmm are the mx versions, but ca and vpd are sc.
       optchi_p14_mx_c3 <- rpmodel:::optimal_chi(kmm_mx, gammastar_mx,
                                                 ns_star_mx, ca_sc, vpd_sc,
                                                 beta = stocker_beta_c3,
                                                 c4 = FALSE)
       optchi_p14_ar_c3 <- rpmodel:::optimal_chi(kmm_ar, gammastar_ar,
                                                 ns_star_ar, ca_ar, vpd_ar,
                                                 beta = stocker_beta_c3,
                                                 c4 = FALSE)

       # rpmodel:::optimal_chi for C4
       optchi_p14_sc_c4 <- rpmodel:::optimal_chi(kmm_sc, gammastar_sc,
                                                 ns_star_sc, ca_sc, vpd_sc,
                                                 beta = stocker_beta_c4,
                                                 c4 = TRUE)
       # The mx version is odd - the pyrealm test uses tc_ar and then
       # scalars for the other inputs to PModelEnvironment, so gammastar
       # ns_star and kmm are the mx versions, but ca and vpd are sc.
       optchi_p14_mx_c4 <- rpmodel:::optimal_chi(kmm_mx, gammastar_mx,
                                                 ns_star_mx, ca_sc, vpd_sc,
                                                 beta = stocker_beta_c4,
                                                 c4 = TRUE)
       optchi_p14_ar_c4 <- rpmodel:::optimal_chi(kmm_ar, gammastar_ar,
                                                 ns_star_ar, ca_ar, vpd_ar,
                                                 beta = stocker_beta_c4,
                                                 c4 = TRUE)


})

# CalcVUEVcmax tests

# NOTE that this testing feeds the optimal chi from both c3 and c4 into
# the three Jmax methods. This isn"t supported by the main rpmodel::rpmodel
# function, which currently (1.0.6) enforces a separate c4 jmax method that
# is identical to using c4 optimal chi with the none Jmax method.

c4 <- c("c3", "c4")

sm <- c("sm-off", "sm-on")

ft <- c("fkphio-off", "fkphio-on")

lue_method <- c("wang17", "smith19", "none")

optchi  <- list(sc = list(kmm = values$kmm_sc,
                          gammastar = values$gammastar_sc,
                          ns_star = values$ns_star_sc,
                          ca = values$ca_sc,
                          vpd = values$vpd_sc),
                ar = list(kmm = values$kmm_ar,
                          gammastar = values$gammastar_ar,
                          ns_star = values$ns_star_ar,
                          ca = values$ca_ar,
                          vpd = values$vpd_ar))

# Needs to match to (reverse) ordering of pytest.mark.parametrise
# variables in pytesting.
luevcmax <- expand.grid(c4 = c4,
                        oc = names(optchi),
                        lm = lue_method,
                        ft = ft,
                        sm = sm,
                        stringsAsFactors = FALSE)

for (rw in seq(nrow(luevcmax))) {

    inputs <- as.list(luevcmax[rw, ])

    oc_vars <- optchi[[inputs$oc]]

    if (inputs$c4 == "c4") {
        c4 <- TRUE
    } else {
        c4 <- FALSE
    }

    if (inputs$ft == "fkphio-off") {
        ftemp_kphio <- 1.0
    } else if (inputs$oc == "sc") {
        ftemp_kphio <- ftemp_kphio(tc = values$tc_sc, c4 = c4)
    } else {
        ftemp_kphio <- ftemp_kphio(tc = values$tc_ar, c4 = c4)
    }

    # Optimal Chi
    this_optchi <- optchi[[inputs$oc]]
    if (c4) {
        this_optchi$beta <- values[['stocker_beta_c4']]
        this_optchi$c4 <- TRUE
        optchi_out <- do.call(rpmodel:::optimal_chi, this_optchi)
    } else {
        this_optchi$beta <- values[['stocker_beta_c3']]
        this_optchi$c4 <- FALSE
        optchi_out <- do.call(rpmodel:::optimal_chi, this_optchi)
    }

    # Soilmstress
    if (inputs$sm == "sm-off") {
        soilmstress <- 1.0
    } else {
        soilmstress <- calc_soilmstress(soilm = values$soilm_sc,
                                        meanalpha = values$meanalpha_sc)
    }

    test_name <- paste0("jmax-", paste(inputs, collapse = "-"))
    values[[test_name]] <- switch(inputs$lm,
        "wang17" = rpmodel:::lue_vcmax_wang17(optchi_out,
                    kphio = ftemp_kphio * values$kphio,
                    soilmstress = soilmstress,
                    c_molmass = values$c_molmass),
        "smith19" = rpmodel:::lue_vcmax_smith19(optchi_out,
                    kphio = ftemp_kphio * values$kphio,
                    soilmstress = soilmstress,
                    c_molmass = values$c_molmass),
        "none" = rpmodel:::lue_vcmax_none(optchi_out,
                    kphio = ftemp_kphio * values$kphio,
                    soilmstress = soilmstress,
                    c_molmass = values$c_molmass))
}


# Rpmodel tests

vars  <- list(sc = list(tc = values$tc_sc,
                        vpd = values$vpd_sc,
                        co2 = values$co2_sc,
                        patm = values$patm_sc,
                        fapar = values$fapar_sc,
                        ppfd = values$ppfd_sc),
              ar = list(tc = values$tc_ar,
                        vpd = values$vpd_ar,
                        co2 = values$co2_ar,
                        patm = values$patm_ar,
                        fapar = values$fapar_ar,
                        ppfd = values$ppfd_ar))

# Needs to match to (reverse) ordering of pytest.mark.parametrise
# variables in pytesting.
rpmodel_c3 <- expand.grid(vr = names(vars),
                          lm = lue_method,
                          ft = ft,
                          sm = sm,
                          stringsAsFactors = FALSE)


run_row <- function(inputs) {

    if (inputs$ft == "fkphio-off") {
        do_ftemp_kphio <- FALSE
    } else {
        do_ftemp_kphio <- TRUE
    }

    # Soilmstress
    if (inputs$sm == "sm-off") {
        do_soilmstress <- FALSE
    } else {
        do_soilmstress <- TRUE
    }

    v <- vars[[inputs$vr]]
    v$kphio <- 0.05
    v$soilm <- values$soilm_sc
    v$meanalpha <- values$meanalpha_sc
    v$do_ftemp_kphio <- do_ftemp_kphio
    v$do_soilmstress <- do_soilmstress
    v$method_jmaxlim <- inputs$lm
    v$c4 <- FALSE
    v$bpar_soilm <- 0.733

    # Run the P model
    ret <- do.call(rpmodel, v)

    # The rpmodel implementation is sensitive to rounding error when
    # using the simple method. fact_jmaxlim can be (just) > 1, leading to
    # sqrt(-tinything) = NaN. This leads to differences in pmodel outputs,
    # so trapping that here.
    ret$jmax <- ifelse(is.nan(ret$jmax), Inf, ret$jmax)

    test_name <- paste("rpmodel-c3", paste(inputs, collapse = "-"), sep = "-")

    return(list(name = test_name, val = ret))
}


for (rw in seq(nrow(rpmodel_c3))) {

    inputs <- as.list(rpmodel_c3[rw, ])
    ret <- run_row(inputs)
    values[[ret$name]] <- ret$val

}

# Repeat for C4 - can"t use different Jmax methods, so drop that

rpmodel_c4 <- expand.grid(vr = names(vars),
                          ft = ft,
                          sm = sm,
                          stringsAsFactors = FALSE)


for (rw in seq(nrow(rpmodel_c4))) {

    inputs <- as.list(rpmodel_c4[rw, ])

    if (inputs$ft == "fkphio-off") {
        do_ftemp_kphio <- FALSE
    } else {
        do_ftemp_kphio <- TRUE
    }

    # Soilmstress
    if (inputs$sm == "sm-off") {
        do_soilmstress <- FALSE
    } else {
        do_soilmstress <- TRUE
    }

    v <- vars[[inputs$vr]]
    v$c4 <- TRUE
    v$kphio <- 0.05
    v$soilm <- values$soilm_sc
    v$meanalpha <- values$meanalpha_sc
    v$do_ftemp_kphio <- do_ftemp_kphio
    v$do_soilmstress <- do_soilmstress

    # NOTE - the default value of bpar_soilm passed into soilmstress
    # by rpmodel is different from the default value of bpar set in the
    # function definition, so standardise that here
    v$bpar_soilm <- 0.733

    ret <- do.call(rpmodel, v)

    # The rpmodel implementation is sensitive to rounding error when
    # using the simple method. fact_jmaxlim can be (just) > 1, leading to
    # sqrt(-tinything) = NaN. This leads to differences in pmodel outputs,
    # so trapping that here.
    ret$jmax <- ifelse(is.nan(ret$jmax), Inf, ret$jmax)

    test_name <- paste("rpmodel-c4",
                       paste(inputs, collapse = "-"), sep = "-")

    values[[test_name]] <- ret


}

# Save values to YAML for use in python tests.
write_json(values, "test_outputs_rpmodel.json",
           digits = 8, pretty = TRUE, auto_unbox = TRUE,
           na = 'null')
