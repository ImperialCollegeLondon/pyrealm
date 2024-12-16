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
    output <- matrix(NA, nrow = aa, ncol = 16, byrow = T)
    colnames(output) <- c(
        "P0", "dD", "D", "H", "fc", "Ac", "Wf", "Ws", "Wss",
        "GPP", "Rm1", "Rm2", "NPP", "turnover", "dWs", "dWfr"
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
            P0[i], dD / 2 * 1000, d, H, fc, Ac, Wf, Ws, Wss, GPP, Rm1, Rm2, NPP1, NPP2, dWs, dWfr
        )
    }

    return(output)
}

# Load alternative plant functional types
pfts <- read.csv("pft_definitions.csv")

# Generate 100 year sequences of plant growth to do fine regression testing
for (pft_idx in seq_len(nrow(pfts))) {
    # Get the PFT
    pft <- as.list(pfts[pft_idx, ])

    # Seperate off the name
    name <- pft[["name"]]
    pft[["name"]] <- NULL

    # Get GPP sequence
    n_years <- 100
    pft[["P0"]] <- rnorm(n_years, mean = 7)
    pft[["year"]] <- seq(n_years)

    tmodel_run <- do.call(tmodel, pft)

    write.csv(tmodel_run,
        sprintf("rtmodel_output_%s.csv", name),
        row.names = FALSE
    )
}

# Generate some simple snapshot predictions for single years to provide
# expectations for varying matrix inputs in unit testing.


dbh <- c(0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
results <- list()

for (pft_idx in seq_len(nrow(pfts))) {
    # Get the PFT
    pft <- as.list(pfts[pft_idx, ])

    # Seperate off the name
    name <- pft[["name"]]
    pft[["name"]] <- NULL
    pft[["P0"]] <- 7 / 0.9
    pft[["year"]] <- 1

    for (dbh_val in dbh) {
        pft[["d"]] <- dbh_val
        tmodel_run <- do.call(tmodel, pft)
        these_res <- data.frame(tmodel_run)
        these_res["pft_name"] <- name
        results <- c(results, list(these_res))
    }
}

results <- do.call(rbind, results)
write.csv(results, "rtmodel_unit_testing.csv", row.names = FALSE)
