library(bigleaf)
library(jsonlite)


# This file contains two sections:
# 1) doctest examples - this section calculate the expectations from the
#     bigleaf package for the examples used in the hygro function docstrings.
# 2) pytest values - this section calculates a wider set of test values for
#    use in pytest validation against the bigleaf outputs.

# Section 1) Doctests

# calc_vp_sat
Esat.slope(21)
#   Esat     Delta
# 1 2.480904 0.152347
Esat.slope(21, "Allen_1998")
#   Esat     Delta
# 1 2.487005 0.1527629
Esat.slope(21, "Alduchov_1996")
#   Esat     Delta
# 1 2.481888 0.1524929

# vp_to_vpd
e.to.VPD(1.9, 21)
# [1] 0.5809042
e.to.VPD(1.9, 21, "Allen_1998")
# [1] 0.5870054

# rh_to_vpd
rH.to.VPD(0.7, 21)
# [1] 0.7442712
rH.to.VPD(0.7, 21, "Allen_1998")
# [1] 0.7461016

# sh_to_vp
q.to.e(0.006, 99.024)

# sh_to_vpd
q.to.VPD(0.006, 21, 99.024)
# [1] 1.529159
q.to.VPD(0.006, 21, 99.024, "Allen_1998")
# [1] 1.535260

# Section 2) pytest data

temp <- seq(0, 30, length = 31)
vp <- seq(1, 200, length = 31)
rh <- seq(0, 1, length = 31)
sh <- seq(0.01, 0.05, length = 31)
patm <- seq(60, 110, length = 31)

output <- list()

output$calc_vp_sat <- list(
    "Allen1998" = Esat.slope(temp, formula = "Allen_1998")$Esat,
    "Alduchov1996" = Esat.slope(temp, formula = "Alduchov_1996")$Esat,
    "Sonntag1990" = Esat.slope(temp, formula = "Sonntag_1990")$Esat
)

output$convert_vp_to_vpd <- list(
    "Allen1998" = e.to.VPD(vp, temp, Esat.formula = "Allen_1998"),
    "Alduchov1996" = e.to.VPD(vp, temp, Esat.formula = "Alduchov_1996"),
    "Sonntag1990" = e.to.VPD(vp, temp, Esat.formula = "Sonntag_1990")
)

output$convert_rh_to_vpd <- list(
    "Allen1998" = rH.to.VPD(rh, temp, Esat.formula = "Allen_1998"),
    "Alduchov1996" = rH.to.VPD(rh, temp, Esat.formula = "Alduchov_1996"),
    "Sonntag1990" = rH.to.VPD(rh, temp, Esat.formula = "Sonntag_1990")
)


output$convert_sh_to_vp <- q.to.e(sh, patm)

output$convert_sh_to_vpd <- list(
    "Allen1998" = q.to.VPD(sh, temp, patm, Esat.formula = "Allen_1998"),
    "Alduchov1996" = q.to.VPD(sh, temp, patm, Esat.formula = "Alduchov_1996"),
    "Sonntag1990" = q.to.VPD(sh, temp, patm, Esat.formula = "Sonntag_1990")
)

jsonlite::write_json(output, "bigleaf_test_values.json", digits = 8)
