# https://github.com/dsval/Environmental-influences-on-the-maximum-quantum-yield-of-terrestrial-primary-production/blob/1135978b745c49b4849b8b3feb22d438d0df6fc4/1_Define_functions.R#L1549-L1627

# calc_phi0 new
calc_phi0_new <- function(tc, mGDD0 = NA, AI) {
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
    phi_o_theo <- 1 / 9 # theoretical maximum phi0 (Long, 1993;Sandoval et al., in.prep.)
    m <- 4.090556 # curvature parameter phio max (Sandoval et al., in.prep.) IN SITU FAPAR!!!!
    n <- 0.121122 # curvature parameter phio max (Sandoval et al., in.prep.)IN SITU FAPAR!!!!
    # m <- 0.4               	# curvature parameter phio max (Sandoval et al., in.prep.) OPTIMIZED FLUX DATA KIT !!!!
    # n <- 1.01           		# curvature parameter phio max (Sandoval et al., in.prep.)OPTIMIZED FLUX DATA KIT !!!!
    Rgas <- 8.3145 # ideal gas constant J/mol/K
    dS0 <- 3468.185 # max entropy change(Sandoval et al., in.prep.)
    dS_mgdd <- 0.6680158 # rate entropy change with temperature phio max (Sandoval et al., in.prep.)
    Ha <- 70885.39 # activation energy J/mol (Sandoval et al., in.prep.)
    # Ha <- 62000     		# activation energy J/mol (Sandoval et al., in.prep.)
    # if mGDD0 is missing, calculate
    if (is.na(mGDD0)) {
        mGDD0 <- mean(tc[tc > 0], na.rm = T)
    }
    ## calc activation entropy, J/mol/K (Sandoval et al., in.prep.)
    # DeltaS = 1558.853-50.223*mGDD0
    # DeltaS = dS0-dS_mgdd*mGDD0
    ## power law from flux data kit
    DeltaS <- dS0 * mGDD0^(-dS_mgdd)
    ## calc deaactivation energy J/mol (Sandoval et al., in.prep.)
    Hd <- 295 * DeltaS

    ###############################################################################################
    # 02.define the functions
    ###############################################################################################

    no_acc_f_arr <- function(tcleaf, Ha = 71513, Hd = 2e+05, dent = 649) {
        ### 10.1111/nph.16883
        Rgas <- 8.3145 # J/mol/K
        ## fix for optimization
        if (!is.na(Ha) & !is.na(Hd) & Ha > Hd) {
            Ha <- Hd - 1
        }
        Top <- Hd / (dent - Rgas * log(Ha / (Hd - Ha)))
        tkleaf <- tcleaf + 273.15
        ################### change to Medlyn et al. (2002)
        f1 <- exp((Ha * (tkleaf - Top)) / (Top * Rgas * tkleaf))
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
