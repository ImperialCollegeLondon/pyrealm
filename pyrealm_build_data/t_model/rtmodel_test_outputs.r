# This file provides a benchmark dataset for T Model implementations using
# the original R implementation of the T Model was provided by Guangqi Li.

# The version of the tmodel function here has been slightly altered from
# the reference function in the t_model.R file to record more of the internal
# variables for validation


tmodel <- function(P0, year, a, cr, Hm, rho, rr,
                   rs, L, zeta, y, sigma, tf, tr, d, K) {
    # P0: potential annual GPP (P model). 10%: foliage matainance respiration
    P0 <- P0 * (1 - 0.1)
    aa <- length(year) # simulate years
    output <- matrix()
    output <- matrix(NA, nrow = aa, ncol = 12, byrow = T)
    colnames(output) <- c(
        "dD", "D", "H", "Ac", "Wf", "Ws", "Wss",
        "GPP", "Rm1", "Rm2", "dWs", "dWfr"
    ) # you can decide which index you want output
    dD <- 0
    NPP1 <- NA
    for (i in 1:aa) {
        # diameter (d) with accumulate each year, and the growth is dD
        d <- d + dD
        # H is controlled by d, and constrained by maximum height (Hm)
        H <- Hm * (1 - exp(-a * d / Hm))
        # crown ratio
        fc <- H / (a * d)
        # dWs <- pi/8 * rho * d * (a * d * (1 - (H/Hm)) + 2 * H)
        # crown area
        Ac <- ((pi * cr) / (4 * a)) * d * H
        # stem mass
        Ws <- (pi / 8) * (d^2) * H * rho
        # foliage mass
        Wf <- Ac * L * (sigma^(-1))
        # sapwood mass
        Wss <- Ac * rho * H * (1 - fc / 2) / cr
        # GPP captured by crown
        GPP <- (Ac * P0[i] * (1 - exp(-(K * L))))
        # GPP fixed per m2 of crown
        gpp <- P0[i] * (1 - exp(-(K * L)))
        # sapwood respiration
        Rm1 <- Wss * rs
        # fine root respiration
        Rm2 <- zeta * sigma * Wf * rr
        # NPP after multiplied by the yeild factor
        NPP1 <- y * (GPP - Rm1 - Rm2)
        # turnover of foliage and fine root
        NPP2 <- (Ac * L * ((1 / (sigma * tf)) + (zeta / tr)))

        num <- y * (
            gpp - rho * (1 - H / (2 * a * d)) * H * rs / cr - L * zeta * rr
        ) - L * (1 / (sigma * tf) + zeta * (1 / tr))

        den <- (a / (2 * cr)) * rho *
            (a * d * (1 / H - 1 / Hm) + 2) +
            (L / d) *
                (a * d * (1 / H - 1 / Hm) + 1) *
                (1 / sigma + zeta)

        # increment of diameter
        dD <- num / den
        # increment of wood
        dWs <- (pi / 8 * rho * d *
            (a * d * (1 - (H / Hm)) + 2 * H)
        ) * dD

        # DO change. This should be identical to NPP3, except for the
        # final scaling to dD but the brackets differ. I've replaced
        # it with code that matches NPP3

        # increament of foliage and fine root
        # dWfr <- (L * ((pi * cr)/(4 * a)) * (a * d * (1 - (H/Hm) + H)) *
        #  (1/sigma + zeta)) * dD

        # increament of foliage and fine root
        dWfr <- (L * (
            (pi * cr) / (4 * a)
        ) * (
            a * d * (1 - H / Hm) + H
        )) * (1 / sigma + zeta) * dD
        output[i, ] <- c(
            dD / 2 * 1000, d, H, Ac, Wf, Ws, Wss, GPP, Rm1, Rm2, dWs, dWfr
        )
    }

    return(output)
}

tmodel_run <- tmodel(rep(7, 100), seq(100),
    d = 0.1,
    a = 116.0,
    cr = 390.43,
    Hm = 25.33,
    rho = 200.0,
    L = 1.8,
    sigma = 14.0,
    tf = 4.0,
    tr = 1.04,
    K = 0.5,
    y = 0.17,
    zeta = 0.17,
    rr = 0.913,
    rs = 0.044
)

write.csv(tmodel_run, "rtmodel_output.csv", row.names = FALSE)
