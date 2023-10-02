# This is the original file provided by Guangqi Li

########## T model part
tmodel <- function(P0, year, a, cr, Hm, rho, rr, rs, L, zeta, y, sigma, tf, tr, d, K) {
    P0 <- P0 * (1 - 0.1) # P0: potential annual GPP (P model). 10%: foliage matainance respiration
    aa <- length(year) # simulate years
    output <- matrix()
    output <- matrix(NA, nrow = aa, ncol = 4, byrow = T)
    colnames(output) <- c("dD", "D", "H", "Ac") # you can decide which index you want output
    dD <- 0
    NPP1 <- NA
    for (i in 1:aa) {
        d <- d + dD # diameter (d) with accumulate each year, and the growth is dD
        H <- Hm * (1 - exp(-a * d / Hm)) # H is controlled by d, and constrained by maximum height (Hm)
        fc <- H / (a * d) # crown ratio
        # 		dWs <- pi/8 * rho * d * (a * d * (1 - (H/Hm)) + 2 * H)
        Ac <- ((pi * cr) / (4 * a)) * d * H # crown area
        Ws <- (pi / 8) * (d^2) * H * rho # stem mass
        Wf <- Ac * L * (sigma^(-1)) # foliage mass
        Wss <- Ac * rho * H * (1 - fc / 2) / cr # sapwood mass
        GPP <- (Ac * P0[i] * (1 - exp(-(K * L)))) # GPP captured by crown
        gpp <- P0[i] * (1 - exp(-(K * L))) # GPP fixed per m2 of crown
        Rm1 <- Wss * rs # sapwood respiration
        Rm2 <- zeta * sigma * Wf * rr # fine root respiration
        NPP1 <- y * (GPP - Rm1 - Rm2) # NPP after multiplied by the yeild factor
        NPP2 <- (Ac * L * ((1 / (sigma * tf)) + (zeta / tr))) # turnover of foliage and fine root
        # 		NPP3 <- (L * ((pi * cr)/(4 * a)) * (a * d * (1 - H/Hm) + H)) * (1/sigma + zeta)
        # 		dD <- (NPP1 - NPP2)/(NPP3 + dWs)
        num <- y * (gpp - rho * (1 - H / (2 * a * d)) * H * rs / cr - L * zeta * rr) - L * (1 / (sigma * tf) + zeta * (1 / tr))
        den <- (a / (2 * cr)) * rho * (a * d * (1 / H - 1 / Hm) + 2) + (L / d) * (a * d * (1 / H - 1 / Hm) + 1) * (1 / sigma + zeta)
        dD <- num / den # increment of diameter
        dWs <- (pi / 8 * rho * d * (a * d * (1 - (H / Hm)) + 2 * H)) * dD # increment of wood
        dWfr <- (L * ((pi * cr) / (4 * a)) * (a * d * (1 - (H / Hm) + H)) * (1 / sigma + zeta)) * dD # increament of foliage and fine root
        output[i, ] <- c(dD / 2 * 1000, d, H, Ac)
    }
    output
}
