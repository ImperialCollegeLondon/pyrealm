# calc_phi0
calc_phi0 <- function(AI, tc, mGDD0 = NA, phi_o_theo = 1 / 9) {
    # ************************************************************************
    # Name:     calc_phi0
    # Inputs:   - double - scalar (AI), climatological aridity index, defined as PET/P
    #           - double - vector (tc), air temperature, degrees C
    #           - double - scalar (mGDD0), mean temperature during growing degree days with tc>0
    # Returns:  double, intrinsic quantum yield at temperature tc, mol CO2/mol photon
    # Features: This function calculates the temperature and aridity dependence of the
    #           Intrinsic quantum Yield
    # * Ref:    Sandoval, Flo, Morfopoulus and Prentice
    # 		    The temperature effect on the intrinsic quantum yield at the ecosystem level
    #             in prep.;
    #             doi:
    # ************************************************************************
    ###############################################################################################
    # 01.define the parameters/constants
    ###############################################################################################

    # DO change here to avoid imprecision in theoretical maxmimum and to allow alternatives
    # phi_o_theo <- 0.111 # theoretical maximum phi0 (Long, 1993;Sandoval et al., in.prep.)

    m <- 6.8681 # curvature parameter phio max (Sandoval et al., in.prep.)
    n <- 0.07956432 # curvature parameter phio max (Sandoval et al., in.prep.)
    Rgas <- 8.3145 # ideal gas constant J/mol/K
    Ha <- 75000 # activation energy J/mol (Sandoval et al., in.prep.)
    # if mGDD0 is missing, calculate
    if (is.na(mGDD0)) {
        mGDD0 <- mean(tc[tc > 0], na.rm = T)
    }
    ## calc activation entropy, J/mol/K (Sandoval et al., in.prep.)
    DeltaS <- 1558.853 - 50.223 * mGDD0
    ## calc deaactivation energy J/mol (Sandoval et al., in.prep.)
    Hd <- 294.804 * DeltaS

    ###############################################################################################
    # 02.define the functions
    ###############################################################################################

    no_acc_f_arr <- function(tcleaf, Ha = 71513, Hd = 2e+05, dent = 649) {
        ### 10.1111/nph.16883

        ## fix for optimization
        if (!is.na(Ha) & !is.na(Hd) & Ha > Hd) {
            Ha <- Hd - 1
        }

        Top <- Hd / (dent - Rgas * log(Ha / (Hd - Ha)))
        tkleaf <- tcleaf + 273.15

        f1 <- (tkleaf / Top) * exp((Ha * (tkleaf - Top)) / (Top * Rgas * tkleaf))
        f2 <- 1 + exp((Top * dent - Hd) / (Top * Rgas))
        f3 <- 1 + exp((tkleaf * dent - Hd) / (tkleaf * Rgas))

        farr <- f1 * (f2 / f3)

        return(farr)
    }
    ###############################################################################################
    # 03.calculate maximum phi0
    ###############################################################################################
    phi_o_peak <- (phi_o_theo / (1 + (AI)^m)^n)
    ###############################################################################################
    # 04.calculate temperature dependence of phi0
    ###############################################################################################
    phi0_fT <- no_acc_f_arr(tcleaf = tc, Ha = Ha, Hd = Hd, dent = DeltaS)
    ###############################################################################################
    # 05.calculate phi0
    ###############################################################################################
    phi0 <- phi_o_peak * phi0_fT
    return(phi0)
}
